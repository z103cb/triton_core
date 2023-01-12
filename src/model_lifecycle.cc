// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "model_lifecycle.h"

#include <algorithm>
#include <deque>
#include <future>
#include <stdexcept>
#include <thread>
#include "constants.h"
#include "filesystem.h"
#include "model.h"
#include "model_config_utils.h"
#include "repo_agent.h"
#include "triton/common/logging.h"
#include "triton/common/thread_pool.h"
#include "server.h"

#include "backend_model.h"
#ifdef TRITON_ENABLE_ENSEMBLE
#include "ensemble_model.h"
#endif  // TRITON_ENABLE_ENSEMBLE

namespace triton { namespace core {

const std::string&
ModelReadyStateString(ModelReadyState state)
{
  switch (state) {
    case ModelReadyState::UNKNOWN: {
      static std::string m("UNKNOWN");
      return m;
    }
    case ModelReadyState::READY: {
      static std::string m("READY");
      return m;
    }
    case ModelReadyState::UNAVAILABLE: {
      static std::string m("UNAVAILABLE");
      return m;
    }
    case ModelReadyState::LOADING: {
      static std::string m("LOADING");
      return m;
    }
    case ModelReadyState::UNLOADING: {
      static std::string m("UNLOADING");
      return m;
    }
  }

  static std::string m("<unknown>");
  return m;
}

namespace {

Status
VersionsToLoad(
    const std::string model_path, const std::string& name,
    const inference::ModelConfig& model_config, std::set<int64_t>* versions)
{
  versions->clear();

  // Get integral number of the version directory
  std::set<std::string> subdirs;
  RETURN_IF_ERROR(GetDirectorySubdirs(model_path, &subdirs));
  std::set<int64_t, std::greater<int64_t>> existing_versions;
  for (const auto& subdir : subdirs) {
    if (subdir == kWarmupDataFolder || subdir == kInitialStateFolder) {
      continue;
    }
    if ((subdir.length() > 1) && (subdir.front() == '0')) {
      LOG_WARNING << "ignore version directory '" << subdir
                  << "' which contains leading zeros in its directory name";
      continue;
    }
    try {
      int64_t version = std::stoll(subdir);
      existing_versions.insert(version);
    }
    catch (const std::invalid_argument& ia) {
      LOG_WARNING << "ignore version directory '" << subdir
                  << "' which fails to convert to integral number";
    }
  }

  if (model_config.version_policy().has_specific()) {
    for (const auto& v : model_config.version_policy().specific().versions()) {
      // Only load the specific versions that are presented in model directory
      bool version_not_exist = existing_versions.insert(v).second;
      if (!version_not_exist) {
        versions->emplace(v);
      } else {
        LOG_ERROR << "version " << v << " is specified for model '" << name
                  << "', but the version directory is not present";
      }
    }
  } else {
    if (model_config.version_policy().has_latest()) {
      // std::set is sorted with std::greater
      for (const auto& v : existing_versions) {
        if (versions->size() >=
            model_config.version_policy().latest().num_versions()) {
          break;
        }
        versions->emplace(v);
      }
    } else {
      // all
      versions->insert(existing_versions.begin(), existing_versions.end());
    }
  }

  return Status::Success;
}

// Use smart pointer with custom deleter so that model state will be updated
// to UNAVAILABLE if all smart pointer copies are out of scope
struct ModelDeleter {
  ModelDeleter(std::function<void()> OnDestroyModel)
      : OnDestroyModel_(std::move(OnDestroyModel))
  {
  }

  void operator()(Model* model)
  {
    // The actual model object must be destroyed in a different
    // thread. This thread could have a callstack that includes the
    // model itself because this deleter could be triggered by
    // a request release or response send in the model. Following
    // delete will lead to the model destructor which may wait on this
    // same thread... so deadlock if we don't use a different thread
    // here.
    std::function<void()> destroy_fn = OnDestroyModel_;
    std::thread dthd([model, destroy_fn]() {
      delete model;
      destroy_fn();
    });

    dthd.detach();
  }

  // Use to inform the ModelLifeCycle that the model handle is destroyed
  std::function<void()> OnDestroyModel_;
};

}  // namespace

Status
ModelLifeCycle::Create(
    InferenceServer* server, const ModelLifeCycleOptions& options,
    std::unique_ptr<ModelLifeCycle>* life_cycle)
{
  std::unique_ptr<ModelLifeCycle> local_life_cycle(
      new ModelLifeCycle(server, options));

  *life_cycle = std::move(local_life_cycle);
  return Status::Success;
}

const ModelStateMap
ModelLifeCycle::LiveModelStates(bool strict_readiness)
{
  LOG_VERBOSE(2) << "LiveModelStates()";
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  ModelStateMap live_model_states;
  for (auto& model_version : map_) {
    bool live = false;
    VersionStateMap version_map;

    for (auto& version_model : model_version.second) {
      std::lock_guard<std::mutex> lock(version_model.second->mtx_);
      if (strict_readiness &&
          version_model.second->state_ != ModelReadyState::READY) {
        continue;
      }

      // At least one version is live (ready / loading / unloading)
      if ((version_model.second->state_ != ModelReadyState::UNKNOWN) &&
          (version_model.second->state_ != ModelReadyState::UNAVAILABLE)) {
        live = true;
        version_map[version_model.first] = std::make_pair(
            version_model.second->state_, version_model.second->state_reason_);
      }
    }

    if (live) {
      live_model_states[model_version.first] = std::move(version_map);
    }
  }
  return live_model_states;
}

Status
ModelLifeCycle::StopAllModels()
{
  LOG_VERBOSE(2) << "StopAllModels()";
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  for (auto& model_version : map_) {
    for (auto& version_model : model_version.second) {
      if (version_model.second != nullptr) {
        std::lock_guard<std::mutex> lock(version_model.second->mtx_);
        if (version_model.second->model_ != nullptr) {
          version_model.second->model_->Stop();
        }
      }
    }
  }
  return Status::Success;
}

const std::set<std::tuple<std::string, int64_t, size_t>>
ModelLifeCycle::InflightStatus()
{
  LOG_VERBOSE(2) << "InflightStatus()";
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  std::set<std::tuple<std::string, int64_t, size_t>> inflight_status;
  for (auto& model_version : map_) {
    for (auto& version_model : model_version.second) {
      if (version_model.second != nullptr) {
        std::lock_guard<std::mutex> lock(version_model.second->mtx_);
        if (version_model.second->model_ != nullptr) {
          const auto cnt =
              version_model.second->model_->InflightInferenceCount();
          if (cnt != 0) {
            inflight_status.emplace(
                model_version.first, version_model.first, cnt);
          }
        }
      }
    }
  }
  return inflight_status;
}

const ModelStateMap
ModelLifeCycle::ModelStates()
{
  LOG_VERBOSE(2) << "ModelStates()";
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  ModelStateMap model_states;
  for (auto& model_version : map_) {
    VersionStateMap version_map;

    for (auto& version_model : model_version.second) {
      std::lock_guard<std::mutex> lock(version_model.second->mtx_);
      version_map[version_model.first] = std::make_pair(
          version_model.second->state_, version_model.second->state_reason_);
    }

    model_states[model_version.first] = std::move(version_map);
  }

  return model_states;
}

const VersionStateMap
ModelLifeCycle::VersionStates(const std::string& model_name)
{
  LOG_VERBOSE(2) << "VersionStates() '" << model_name << "'";
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  VersionStateMap version_map;
  auto mit = map_.find(model_name);
  if (mit != map_.end()) {
    for (auto& version_model : mit->second) {
      std::lock_guard<std::mutex> lock(version_model.second->mtx_);
      version_map[version_model.first] = std::make_pair(
          version_model.second->state_, version_model.second->state_reason_);
    }
  }

  return version_map;
}

Status
ModelLifeCycle::ModelState(
    const std::string& model_name, const int64_t model_version,
    ModelReadyState* state)
{
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  auto mit = map_.find(model_name);
  if (mit != map_.end()) {
    auto vit = mit->second.find(model_version);
    if (vit != mit->second.end()) {
      std::lock_guard<std::mutex> lock(vit->second->mtx_);
      *state = vit->second->state_;
      return Status::Success;
    }
  }

  return Status(
      Status::Code::NOT_FOUND, "model '" + model_name + "', version " +
                                   std::to_string(model_version) +
                                   " is not found");
}

Status
ModelLifeCycle::GetModel(
    const std::string& model_name, const int64_t version,
    std::shared_ptr<Model>* model)
{
  LOG_VERBOSE(2) << "GetModel() '" << model_name << "' version " << version;
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  auto mit = map_.find(model_name);
  if (mit == map_.end()) {
    return Status(Status::Code::NOT_FOUND, "'" + model_name + "' is not found");
  }

  auto vit = mit->second.find(version);
  if (vit == mit->second.end()) {
    if (version != -1) {
      return Status(
          Status::Code::NOT_FOUND, "'" + model_name + "' version " +
                                       std::to_string(version) +
                                       " is not found");
    }

    // The case where the request is asking for latest version
    int64_t latest = -1;
    for (auto& version_model : mit->second) {
      if (version_model.first > latest) {
        std::lock_guard<std::mutex> lock(version_model.second->mtx_);
        if (version_model.second->state_ == ModelReadyState::READY) {
          latest = version_model.first;
          // Tedious, but have to set handle for any "latest" version
          // at the moment to avoid edge case like the following:
          // "versions : 1 3 2", version 3 is latest but is requested
          // to be unloaded when the iterator is examining version 2,
          // then 'model' will ensure version 3 is still valid
          *model = version_model.second->model_;
        }
      }
    }
    if (latest == -1) {
      return Status(
          Status::Code::NOT_FOUND,
          "'" + model_name + "' has no available versions");
    }
  } else {
    std::lock_guard<std::mutex> lock(vit->second->mtx_);
    if (vit->second->state_ == ModelReadyState::READY) {
      *model = vit->second->model_;
    } else {
      return Status(
          Status::Code::UNAVAILABLE, "'" + model_name + "' version " +
                                         std::to_string(version) +
                                         " is not at ready state");
    }
  }
  return Status::Success;
}

Status
ModelLifeCycle::AsyncUnload(const std::string& model_name)
{
  LOG_VERBOSE(2) << "AsyncUnload() '" << model_name << "'";
  std::lock_guard<std::mutex> map_lock(map_mtx_);
  auto it = map_.find(model_name);
  if (it == map_.end()) {
    return Status(
        Status::Code::INVALID_ARG, "Model to be unloaded has not been served");
  }

  // Get the existing agent models and notify the unload action
  const uint64_t now_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  for (auto& version : it->second) {
    auto& model_info = version.second;
    std::lock_guard<std::mutex> lock(model_info->mtx_);
    model_info->last_update_ns_ = now_ns;
    // Unload serving model, for model that is in LOADING state,
    // the updated timestamp will be recognized that there is newer update
    // on the model info and the load should be aborted
    if (model_info->state_ == ModelReadyState::READY) {
      if (model_info->agent_model_list_ != nullptr) {
        // Only log the error because the model should be unloaded regardless
        auto status = model_info->agent_model_list_->InvokeAgentModels(
            TRITONREPOAGENT_ACTION_UNLOAD);
        if (!status.IsOk()) {
          LOG_ERROR
              << "Agent model returns error on TRITONREPOAGENT_ACTION_UNLOAD: "
              << status.AsString();
        }
      }

      // unload
      model_info->Release();
    }
  }

  return Status::Success;
}

Status
ModelLifeCycle::AsyncLoad(
    const std::string& model_name, const std::string& model_path,
    const inference::ModelConfig& model_config, const bool is_config_provided,
    const std::shared_ptr<TritonRepoAgentModelList>& agent_model_list,
    std::function<void(Status)>&& OnComplete)
{
  LOG_VERBOSE(2) << "AsyncLoad() '" << model_name << "'";

  std::lock_guard<std::mutex> map_lock(map_mtx_);
  auto it = map_.find(model_name);
  if (it == map_.end()) {
    it = map_.emplace(std::make_pair(model_name, VersionMap())).first;
  }

  std::set<int64_t> versions;
  RETURN_IF_ERROR(
      VersionsToLoad(model_path, model_name, model_config, &versions));
  if (versions.empty()) {
    return Status(
        Status::Code::INVALID_ARG,
        "at least one version must be available under the version policy of "
        "model '" +
            model_name + "'");
  }


  const uint64_t now_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  std::shared_ptr<LoadTracker> load_tracker(
      new LoadTracker(versions.size(), now_ns));
  for (const auto& version : versions) {
    std::unique_ptr<ModelInfo> linfo(
        new ModelInfo(model_path, model_config, now_ns));
    ModelInfo* model_info = linfo.get();

    LOG_INFO << "loading: " << model_name << ":" << version;
    model_info->state_ = ModelReadyState::LOADING;
    model_info->state_reason_.clear();
    model_info->agent_model_list_ = agent_model_list;

    auto res = it->second.emplace(
        std::make_pair(version, std::unique_ptr<ModelInfo>()));
    if (res.second) {
      res.first->second = std::move(linfo);
    } else {
      // There is already a record of this model version. Check if the version
      // model is being served, if so, the re-load of the version
      // should be performed in background to avoid version downtime.
      // Otherwise, swap and monitor state for newly loading model.
      auto& serving_model = res.first->second;
      std::lock_guard<std::mutex> lock(serving_model->mtx_);
      if (serving_model->state_ == ModelReadyState::READY) {
        background_models_[(uintptr_t)model_info] = std::move(linfo);
      } else {
        // swap the monitoring model info
        serving_model.swap(linfo);

        // further check the state, put to 'background_models_' to keep
        // the object valid if the model is LOADING / UNLOADING, because
        // the model info will be accessed by a different thread once the
        // operation is completed
        if ((linfo->state_ == ModelReadyState::LOADING) ||
            (linfo->state_ == ModelReadyState::UNLOADING)) {
          ModelInfo* key = linfo.get();
          background_models_[(uintptr_t)key] = std::move(linfo);
        }
      }
    }

    // Load model asynchronously via thread pool
    load_pool_->Enqueue([this, model_name, version, model_info, OnComplete,
                         load_tracker, is_config_provided]() {
      CreateModel(model_name, version, model_info, is_config_provided);
      OnLoadComplete(model_name, version, model_info, OnComplete, load_tracker);
    });
  }

  return Status::Success;
}

void
ModelLifeCycle::CreateModel(
    const std::string& model_name, const int64_t version, ModelInfo* model_info,
    const bool is_config_provided)
{
  LOG_VERBOSE(2) << "CreateModel() '" << model_name << "' version " << version;
  const auto& model_config = model_info->model_config_;

  // Create model
  Status status;
  std::unique_ptr<Model> is;

  // If 'backend' is specified in the config then use the new triton
  // backend.
  if (!model_config.backend().empty()) {
    std::unique_ptr<TritonModel> model;
    status = TritonModel::Create(
        server_, model_info->model_path_, cmdline_config_map_, host_policy_map_,
        model_name, version, model_config, is_config_provided, &model);
    is.reset(model.release());
  } else {
#ifdef TRITON_ENABLE_ENSEMBLE
    if (model_info->is_ensemble_) {
      status = EnsembleModel::Create(
          server_, model_info->model_path_, version, model_config,
          is_config_provided, min_compute_capability_, &is);
      // Complete label provider with label information from involved models
      // Must be done here because involved models may not be able to
      // obtained from server because this may happen during server
      // initialization.
      if (status.IsOk()) {
        std::set<std::string> no_label_outputs;
        const auto& label_provider = is->GetLabelProvider();
        for (const auto& output : model_config.output()) {
          if (label_provider->GetLabel(output.name(), 0).empty()) {
            no_label_outputs.emplace(output.name());
          }
        }
        for (const auto& element : model_config.ensemble_scheduling().step()) {
          for (const auto& pair : element.output_map()) {
            // Found model that produce one of the missing output
            if (no_label_outputs.find(pair.second) != no_label_outputs.end()) {
              std::shared_ptr<Model> model;
              // Safe to obtain model because the ensemble can't be loaded
              // until the involved models are ready
              GetModel(element.model_name(), element.model_version(), &model);
              label_provider->AddLabels(
                  pair.second,
                  model->GetLabelProvider()->GetLabels(pair.first));
            }
          }
        }
      }
    } else
#endif  // TRITON_ENABLE_ENSEMBLE
    {
      status = Status(
          Status::Code::INVALID_ARG,
          "unknown platform '" + model_config.platform() + "'");
    }
  }

  std::lock_guard<std::mutex> lock(model_info->mtx_);
  if (status.IsOk()) {
    // [FIXME] better way to manage agent model lifecycle
    // Let the deleter also holds a shared pointer copy of agent model list,
    // because the reference in ModelInfo can be cleared before the Model object
    // is destroyed, and we want agent model to be valid for receiving
    // UNLOAD_COMPLETE signal (see ~TritonRepoAgentModelList for detail)
    auto agent_model_list = model_info->agent_model_list_;
    model_info->model_.reset(
        is.release(), ModelDeleter([this, model_name, version, model_info,
                                    agent_model_list]() mutable {
          LOG_VERBOSE(2) << "OnDestroy callback() '" << model_name
                         << "' version " << version;
          LOG_INFO << "successfully unloaded '" << model_name << "' version "
                   << version;
          // Update model state as it is fully unloaded
          {
            std::lock_guard<std::mutex> lock(model_info->mtx_);
            model_info->state_ = ModelReadyState::UNAVAILABLE;
            model_info->state_reason_ = "unloaded";
          }

          // Check if the model info is in background, if so, remove from the
          // map
          std::lock_guard<std::mutex> lk(this->map_mtx_);
          auto it = this->background_models_.find((uintptr_t)model_info);
          if (it != this->background_models_.end()) {
            this->background_models_.erase(it);
          }
        }));
  } else {
    LOG_ERROR << "failed to load '" << model_name << "' version " << version
              << ": " << status.AsString();
    model_info->state_ = ModelReadyState::UNAVAILABLE;
    model_info->state_reason_ = status.AsString();
  }
}

void
ModelLifeCycle::OnLoadComplete(
    const std::string& model_name, const int64_t version, ModelInfo* model_info,
    std::function<void(Status)> OnComplete,
    std::shared_ptr<LoadTracker> load_tracker)
{
  std::lock_guard<std::mutex> tracker_lock(load_tracker->mtx_);
  ++load_tracker->completed_version_cnt_;
  load_tracker->load_set_[version] = model_info;
  // Version will not be marked ready until all versions are
  // ready, this simplify the unloading when one version fails to load as
  // all other versions won't have inflight requests
  if (model_info->state_ != ModelReadyState::LOADING) {
    load_tracker->load_failed_ = true;
    load_tracker->reason_ +=
        ("version " + std::to_string(version) + " is at " +
         ModelReadyStateString(model_info->state_) +
         " state: " + model_info->state_reason_ + ";");
  }
  // Check if all versions are completed and finish the load
  if (load_tracker->completed_version_cnt_ ==
      load_tracker->affected_version_cnt_) {
    // hold 'map_mtx_' as there will be change onto the model info map
    std::lock_guard<std::mutex> map_lock(map_mtx_);
    auto it = map_.find(model_name);
    // Check if the load is the latest frontground action on the model
    for (const auto& version_info : it->second) {
      if (version_info.second->last_update_ns_ >
          load_tracker->last_update_ns_) {
        load_tracker->load_failed_ = true;
        load_tracker->reason_ =
            "Newer operation has been applied to the model lifecycle, current "
            "load operation is out-dated.";
        break;
      }
    }

    if (load_tracker->load_failed_) {
      // Move agent list out of ModelInfo as it needs to be invoked
      // after all ModelInfos are reset
      std::shared_ptr<TritonRepoAgentModelList> lagent_list;
      if (model_info->agent_model_list_) {
        lagent_list = std::move(model_info->agent_model_list_);
      }
      // If any of the versions fails to load, abort the load and unload
      // all newly loaded versions
      for (auto& loaded : load_tracker->load_set_) {
        // Unload directly, the object is being managed either in frontground
        // or background
        std::lock_guard<std::mutex> lock(loaded.second->mtx_);
        if (loaded.second->model_ != nullptr) {
          loaded.second->Release();
        }
      }

      if (lagent_list) {
        auto status =
            lagent_list->InvokeAgentModels(TRITONREPOAGENT_ACTION_LOAD_FAIL);
        if (!status.IsOk()) {
          LOG_ERROR << "Agent model returns error on "
                       "TRITONREPOAGENT_ACTION_LOAD_FAIL: "
                    << status.AsString();
        }
      }
    } else {
      // Unload any previous loaded versions that are still available
      for (auto& version_info : it->second) {
        auto& mi = version_info.second;
        std::lock_guard<std::mutex> info_lk(mi->mtx_);
        if ((mi->state_ == ModelReadyState::READY) &&
            (mi->last_update_ns_ < load_tracker->last_update_ns_)) {
          if (mi->agent_model_list_ != nullptr) {
            auto status = mi->agent_model_list_->InvokeAgentModels(
                TRITONREPOAGENT_ACTION_UNLOAD);
            if (!status.IsOk()) {
              LOG_ERROR << "Agent model returns error on "
                           "TRITONREPOAGENT_ACTION_UNLOAD: "
                        << status.AsString();
            }
          }

          mi->Release();
        }
      }

      // Mark current versions ready and track info in foreground
      for (auto& loaded : load_tracker->load_set_) {
        std::lock_guard<std::mutex> curr_info_lk(loaded.second->mtx_);
        loaded.second->state_ = ModelReadyState::READY;
        model_info->state_reason_.clear();
        LOG_INFO << "successfully loaded '" << model_name << "' version "
                 << version;

        auto bit = background_models_.find((uintptr_t)loaded.second);
        // Check if the version model is loaded in background, if so,
        // replace and unload the current serving version
        if (bit != background_models_.end()) {
          auto vit = it->second.find(loaded.first);

          // Need to lock the previous model info for in case the model is
          // loading / unloading, this ensure the model state is consistent
          // even when the load / unload is completed.
          std::lock_guard<std::mutex> prev_info_lk(vit->second->mtx_);

          // swap previous info into local unique pointer
          auto linfo = std::move(bit->second);
          vit->second.swap(linfo);
          background_models_.erase(bit);

          // if previous info is under change, put into 'background_models_'
          if ((linfo->state_ == ModelReadyState::LOADING) ||
              (linfo->state_ == ModelReadyState::UNLOADING)) {
            ModelInfo* key = linfo.get();
            background_models_[(uintptr_t)key] = std::move(linfo);
          }
        }
      }
      if (model_info->agent_model_list_) {
        auto status = model_info->agent_model_list_->InvokeAgentModels(
            TRITONREPOAGENT_ACTION_LOAD_COMPLETE);
        if (!status.IsOk()) {
          LOG_ERROR << "Agent model returns error on "
                       "TRITONREPOAGENT_ACTION_LOAD_COMPLETE: "
                    << status.AsString();
        }
      }
    }
    if (OnComplete != nullptr) {
      OnComplete(
          load_tracker->load_failed_
              ? Status(Status::Code::INVALID_ARG, load_tracker->reason_)
              : Status::Success);
    }
  }
}

Status
ModelLifeCycle::AddDeleteModelInstances(
  const std::string &model_name,
  const inference::ModelConfig& model_config) 
{
  LOG_INFO << "Starting AddDeleteModelInstances...";

  LOG_INFO << "Finding the model name in the map";
  // Because the instance group API doesn't allow for version specific parameters,
  // we need to iterate through all versions of the model and update their instances 
  // accordingly.
  auto version_map_itr = map_.find(model_name);
  if (version_map_itr == map_.end()) {
    return Status(
      Status::Code::INTERNAL,
      std::string("Did not find model name '") + model_name + std::string("' in model map"));
  }

  LOG_INFO << "Rebuilding the model version map";
  //  Need to iterate through all the versions as there are no version specific
  // settings for the model instances.
  auto& version_map = version_map_itr->second;
  for (auto& model_version : version_map) {
    // This version of the model didn't load and we detected a short
    // circuit. Therefore the user hasn't provided anything which 
    // would change the state of the model failure. Skip this model version.
    std::unique_ptr<ModelInfo>& current_info = model_version.second;
    if (current_info->state_ != triton::core::ModelReadyState::READY) {
      LOG_INFO << "Not modifying the model instances for model '" << model_name << "' version '"
                << std::to_string(model_version.first) << "' due to state " << ModelReadyStateString(current_info->state_)
                << " with reason: " << current_info->state_reason_;
      continue;
    }

    // FIXME: This is a risky downcast. All Models are TritonModels except for
    // Ensemble Models. However, Ensemble Models are not allowed to have instances.
    // Therefore, we are guaranteed the Model can be downcasted to a TritonModel. 
    // This will break if there are any other derived Models which use Model
    // as the Base
    //
    // KMTODO: Assert that only TritonModels are allowed here. 
    LOG_INFO << "Down casting Model to TritonModel";
    TritonModel* raw_triton_model = dynamic_cast<TritonModel*>(current_info->model_.get());
    if (raw_triton_model == nullptr) {
      return Status(
        Status::Code::INTERNAL,
        std::string("Unable to downcast Model to TritonModel for model name: ") + model_name);

    }

    // Collect everything we need to create and destroy instances.
    LOG_INFO << "Collecting all objects needed for instances";
    auto& triton_instance_group_map = raw_triton_model->InstanceGroups();
    auto rate_limiter = raw_triton_model->Server()->GetRateLimiter();
    const triton::common::BackendCmdlineConfigMap& backend_cmdline_config_map = cmdline_config_map_;
    const triton::common::HostPolicyCmdlineConfigMap& host_policy_map = host_policy_map_;
    bool device_blocking = false;
    if (raw_triton_model->Backend()->ExecutionPolicy() ==
        TRITONBACKEND_EXECUTION_DEVICE_BLOCKING) {
      if (model_config.has_sequence_batching()) {
        LOG_INFO << "Overriding execution policy to "
                    "\"TRITONBACKEND_EXECUTION_BLOCKING\" for sequence model \""
                << model_config.name() << "\"";
      } else {
        device_blocking = true;
      }
    }

    // We have the new model configuration but we haven't set it yet.
    LOG_INFO << "Setting updated model config on the model";
    std::string model_config_json;
    RETURN_IF_ERROR(ModelConfigToJson(model_config, 1 /* config version */, &model_config_json));

    TRITONSERVER_Message* message = reinterpret_cast<TRITONSERVER_Message*>(
      new triton::core::TritonServerMessage(
        {model_config_json.c_str(), model_config_json.size()}
        )
      );

    // nocheckin
    const char* msg_ptr;
    size_t msg_size;
    RETURN_IF_TRITONSERVER_ERROR(TRITONSERVER_MessageSerializeToJson(message, &msg_ptr, &msg_size));
    LOG_INFO << "Message to update model config with is: " << msg_ptr;

    // RETURN_IF_ERROR(raw_triton_model->UpdateModelConfig(
    //     1 /* config_version */, message));
    RETURN_IF_ERROR(raw_triton_model->SetModelConfig(model_config));
    RETURN_IF_TRITONSERVER_ERROR(TRITONSERVER_MessageDelete(message));
    
    LOG_INFO << "Rebuilding the device id to backend thread map";
    // Need to rebuild the device_id to BackendThread map since that work was thrown away after
    // loading the model.
    std::map<uint32_t, std::shared_ptr<triton::core::TritonModelInstance::TritonBackendThread>> device_to_thread_map;
    for (const auto& pair : triton_instance_group_map) {
      for (auto& instance : pair.second) {
        // Add to the device_id map
        int32_t device_id = instance->DeviceId();
        auto backend_thread_ptr = instance->BackendThread();
        device_to_thread_map[device_id] = backend_thread_ptr;
      }
    }

    // For each instance group we must add or remove instances.
    // Assumption is the instance groups are in the exact same order 
    // between the old config and the new config. Therefore we should 
    // be able to pick out the names of the groups from the instance_group_map_
    LOG_INFO << "Iterating through model instance groups";
    for (int32_t instance_group_index = 0; instance_group_index < model_config.instance_group().size(); ++instance_group_index) {
      LOG_INFO << "Working with instance group " << std::to_string(instance_group_index);
      const inference::ModelInstanceGroup& instance_group = model_config.instance_group()[instance_group_index];
      std::vector<std::unique_ptr<triton::core::TritonModelInstance>>& triton_instance_group = triton_instance_group_map[instance_group.name()];

      //nocheckin
      LOG_INFO << "Looking for " << instance_group.name();
      for (const auto& instance_group_pair : triton_instance_group_map) {
        LOG_INFO << "key: " << instance_group_pair.first;
      }

      // reasonable assumption that there are less than 2^31 instances of a model to prevent underflow.
      int32_t difference = int32_t(instance_group.count()) - int32_t(triton_instance_group.size());
      LOG_INFO << "instance count difference is: " << std::to_string(instance_group.count()) << " - " << std::to_string(triton_instance_group.size()) << " = " << std::to_string(difference);
      if (difference > 0) {
        // Add instances to the model
        for (int32_t i = 0; i < difference; ++i){ 
          int32_t instance_index = triton_instance_group.size() + i;
          LOG_INFO << "Adding instance as index " << std::to_string(instance_index);

          // Create the model instance
          // The BackendThread is created in creation of the instance
          // This instance gets registered with the rate limiter on creation
          // Instance gets added to the model on creation.
          RETURN_IF_ERROR(TritonModelInstance::CreateOneInstance(
              raw_triton_model, model_config.instance_group()[instance_group_index], backend_cmdline_config_map, host_policy_map,
              device_to_thread_map, device_blocking, instance_index));
        }
      } else if (difference < 0) {
        // Remove instances from the model. Leave minimum of 1
        if ((difference * -1) >= int32_t(triton_instance_group.size())) {
          difference = int32_t(triton_instance_group.size()) - 1;
          difference *= -1;
          LOG_INFO << "Difference in instance count greater than or equals the number of current instances. " 
                  << "Incrementing the difference to leave 1 instance running."
                  << "Number of current instances: " << std::to_string(triton_instance_group.size())
                  << ", difference: " << std::to_string(difference);

          // Need to update the config as well 
          // INCONSISTENT: The model config on the model_infos is const and 
          // apparently at this point we have already updated the model 
          // config on the TritonModel, so modifying here works.
          inference::ModelConfig& triton_model_config = raw_triton_model->MutableConfig();
          google::protobuf::RepeatedPtrField< inference::ModelInstanceGroup >* group_repeated_ptr_ptr = triton_model_config.mutable_instance_group();
          inference::ModelInstanceGroup* triton_model_instance_group = group_repeated_ptr_ptr->Mutable(instance_group_index);
          triton_model_instance_group->set_count(1);
        }
        for (int i = difference; i < 0; ++i) {
          LOG_INFO << "Removing instance as index " << std::to_string(i);
          // BackendThread stops when the TritonModelInstance desturctor is called
          auto& triton_model_instance = triton_instance_group.back();

          // Unregister the TritonModelInstance with the rate_limiter.
          LOG_INFO << "Unregistering model instance from rate_limiter";
          rate_limiter->UnregisterModelInstance(triton_model_instance.get());
 
          // remove the last one in the list for no reason.
          // Remove the instance from the model
          // Backend Thread gets stopped on destruction of the instance.
          triton_instance_group.erase(triton_instance_group.end()-1);

        }
      } // else do nothing
    }
  }
  
  LOG_INFO << "Done with AddDeleteModelInstances...";
  return Status::Success;
}

}}  // namespace triton::core
