// Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "sequence_state.h"

#include "cuda_utils.h"
#include "memory.h"
#include "triton/common/logging.h"

namespace triton { namespace core {

SequenceState::SequenceState() : data_(new MemoryReference) {}

SequenceState::SequenceState(
    const std::string& name, const inference::DataType datatype,
    const int64_t* shape, const uint64_t dim_count, bool reuse_buffer,
    bool use_growable_memory)
    : name_(name), datatype_(datatype), shape_(shape, shape + dim_count),
      data_(new MemoryReference), reuse_buffer_(reuse_buffer),
      use_growable_memory_(use_growable_memory)
{
}

SequenceState::SequenceState(
    const std::string& name, const inference::DataType datatype,
    const std::vector<int64_t>& shape, bool reuse_buffer,
    bool use_growable_memory)
    : name_(name), datatype_(datatype), shape_(shape),
      data_(new MemoryReference), reuse_buffer_(reuse_buffer),
      use_growable_memory_(use_growable_memory)
{
}

Status
SequenceState::SetData(const std::shared_ptr<Memory>& data)
{
  if (data_->TotalByteSize() != 0) {
    return Status(
        Status::Code::INVALID_ARG,
        "state '" + name_ + "' already has data, can't overwrite");
  }

  data_ = data;
  return Status::Success;
}

Status
SequenceState::RemoveAllData()
{
  data_ = std::make_shared<MemoryReference>();
  return Status::Success;
}

Status
SequenceState::SetStringDataToZero()
{
  if (Data()->TotalByteSize() % 4 != 0) {
    return Status(
        Status::Code::INVALID_ARG,
        "The total byte size must be a multiple of 4 when setting the "
        "sequence state to zero.");
  }

  TRITONSERVER_MemoryType memory_type;
  int64_t memory_type_id;

  const std::shared_ptr<MutableMemory>& memory =
      reinterpret_cast<const std::shared_ptr<MutableMemory>&>(Data());
  char* buffer = memory->MutableBuffer(&memory_type, &memory_type_id);
  if (memory_type == TRITONSERVER_MEMORY_GPU) {
    memset(buffer, 0, Data()->TotalByteSize());
  } else {
    RETURN_IF_CUDA_ERR(
        cudaMemset(buffer, 0, Data()->TotalByteSize()),
        std::string("failed to set the data to zero."));
  }

  return Status::Success;
}

Status
SequenceStates::Initialize(
    const std::unordered_map<
        std::string, const inference::ModelSequenceBatching_State&>&
        state_output_config_map,
    const size_t max_batch_size,
    const std::unordered_map<std::string, InitialStateData>& initial_state,
    TRITONSERVER_InstanceGroupKind kind, int32_t device_id)
{
  input_states_.clear();
  output_states_.clear();

  for (auto& state : state_output_config_map) {
    auto& state_config = state.second;
    bool use_single_buffer = state_config.use_single_buffer();
    bool use_growable_memory = state_config.use_growable_memory();

    if (use_growable_memory) {
      if (!use_single_buffer) {
        return Status(
            Status::Code::INVALID_ARG,
            std::string("In order to 'use_growable_memory', you must also "
                        "enable 'use_single_buffer' for state input name '") +
                state_config.input_name() + ("'."));
      }
      if (kind != TRITONSERVER_INSTANCEGROUPKIND_GPU) {
        return Status(
            Status::Code::INVALID_ARG,
            std::string("'use_growable_memory' can only work with KIND_GPU "
                        "model instances right now. Please either change the "
                        "instance type to KIND_GPU or set "
                        "'use_growable_memory' to false for state input '") +
                state_config.input_name() + ("'."));
      }
    }

    std::vector<int64_t> dims;
    if (max_batch_size != 0) {
      dims.push_back(1);
    }

    // Convert the variable dimensions to 1 for the first request.
    for (auto& dim : state_config.dims()) {
      if (dim == -1) {
        dims.push_back(1);
      } else {
        dims.push_back(dim);
      }
    }

    std::shared_ptr<MutableMemory> data;
    auto initial_state_it = initial_state.find(state_config.input_name());
    if (initial_state_it != initial_state.end()) {
      if (use_growable_memory) {
        std::unique_ptr<GrowableMemory> growable_memory;
        RETURN_IF_ERROR(GrowableMemory::Create(
            initial_state_it->second.data_->TotalByteSize(),
            TRITONSERVER_MEMORY_GPU, device_id, growable_memory,
            1024 * 1024 * 1024));
        data = std::move(growable_memory);

        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        char* dst_buffer = data->MutableBuffer(&memory_type, &memory_type_id);
        RETURN_IF_CUDA_ERR(
            cudaMemcpy(
                growable_memory->MutableBuffer(&memory_type, &memory_type_id),
                dst_buffer, initial_state_it->second.data_->TotalByteSize(),
                cudaMemcpyHostToDevice),
            std::string("failed to copy initial state to GPU:"));
      } else {
        data = std::make_shared<AllocatedMemory>(
            initial_state_it->second.data_->TotalByteSize(),
            TRITONSERVER_MEMORY_CPU, 0);

        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        char* dst_buffer = data->MutableBuffer(&memory_type, &memory_type_id);
        char* initial_state_buffer =
            initial_state_it->second.data_->MutableBuffer(
                &memory_type, &memory_type_id);

        memcpy(
            dst_buffer, initial_state_buffer,
            initial_state_it->second.data_->TotalByteSize());
      }
    } else {
      size_t state_size;
      if (state.second.data_type() == inference::DataType::TYPE_STRING) {
        auto element_count = triton::common::GetElementCount(dims);
        // Total number of bytes required is equal to the element count
        // multiplied by 4.
        state_size = 4 * element_count;
      } else {
        state_size =
            triton::common::GetByteSize(state.second.data_type(), dims);
      }
      if (use_growable_memory) {
        data = std::make_shared<AllocatedMemory>(
            state_size, TRITONSERVER_MEMORY_CPU, 0);
      } else {
        std::unique_ptr<GrowableMemory> growable_memory;
        RETURN_IF_ERROR(GrowableMemory::Create(
            state_size, TRITONSERVER_MEMORY_GPU, device_id, growable_memory,
            1024 * 1024 * 1024));
        data = std::move(growable_memory);
      }
    }

    const auto& input_pair = input_states_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(state_config.input_name()),
        std::forward_as_tuple(new SequenceState(
            state_config.input_name(), state.second.data_type(), dims,
            use_single_buffer, use_growable_memory)));

    if (!input_pair.second) {
      LOG_WARNING
          << "Detected duplicate 'input_name' in the state configuration: '"
          << state_config.input_name()
          << ".' This state configuration will be ignored.";
      continue;
    }

    auto& input_tensor = input_pair.first->second;
    RETURN_IF_ERROR(input_tensor->SetData(data));
    if (input_tensor->DType() == inference::DataType::TYPE_STRING) {
      RETURN_IF_ERROR(input_tensor->SetStringDataToZero());
    }

    const auto& output_pair = output_states_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(state_config.output_name()),
        std::forward_as_tuple(new SequenceState(
            state_config.output_name(), state.second.data_type(), dims,
            use_single_buffer, use_growable_memory)));
    if (!output_pair.second) {
      // Remove the corresponding state from the input_states_map
      input_states_.erase(state_config.input_name());
      LOG_WARNING << "Detected duplicate 'output_name' in the state "
                     "configuration: '"
                  << state_config.output_name()
                  << "'. This state configuration will be ignored.";

      continue;
    }

    auto& output_tensor = output_pair.first->second;
    if (use_single_buffer) {
      auto& output_tensor = output_pair.first->second;
      output_tensor->SetData(data);
    }

    output_tensor->SetOtherState(input_tensor);
    input_tensor->SetOtherState(output_tensor);
  }

  return Status::Success;
}

Status
SequenceStates::OutputState(
    const std::string& name, const inference::DataType datatype,
    const int64_t* shape, const uint64_t dim_count,
    SequenceState** output_state)
{
  const auto& output_state_itr = output_states_.find(name);

  // If the state name is not valid return an error.
  if (output_state_itr == output_states_.end()) {
    return Status(
        Status::Code::INVALID_ARG,
        "state '" + name + "' is not a valid state name.");
  }

  std::vector<int64_t> new_shape(shape, shape + dim_count);
  *output_states_[name]->MutableDType() = datatype;
  *output_states_[name]->MutableShape() = new_shape;

  auto& output_state_r = output_states_[name];
  size_t iter_advance =
      std::distance(output_states_.begin(), output_states_.find(name));

  // Find the input state corresponding to this output state.
  auto input_states_itr = input_states_.begin();
  std::advance(input_states_itr, iter_advance);
  auto& input_state_r = input_states_[input_states_itr->first];
  bool reuse_buffer = output_states_[name]->ReuseBuffer();

  if (output_state != nullptr) {
    *output_state = output_states_[name].get();
  }

  output_state_r->SetStateUpdateCallback(
      [&output_state_r, &input_state_r, reuse_buffer]() {
        // Swap the internal memory if the size of the input and output state is
        // equal

        if (output_state_r->Data()->TotalByteSize() ==
            input_state_r->Data()->TotalByteSize()) {
          if (!reuse_buffer) {
            std::shared_ptr<Memory> temp_memory = input_state_r->Data();
            RETURN_IF_ERROR(input_state_r->RemoveAllData());
            RETURN_IF_ERROR(input_state_r->SetData(output_state_r->Data()));
            RETURN_IF_ERROR(output_state_r->RemoveAllData());
            RETURN_IF_ERROR(output_state_r->SetData(temp_memory));
          }
        } else {
          if (!reuse_buffer) {
            // If the size of output state is different from the input state,
            // allocate a new memory for the input state with the same size as
            // output state.
            TRITONSERVER_MemoryType memory_type;
            int64_t memory_type_id;

            const std::shared_ptr<MutableMemory>& input_memory =
                reinterpret_cast<const std::shared_ptr<MutableMemory>&>(
                    input_state_r->Data());

            input_memory->MutableBuffer(&memory_type, &memory_type_id);
            std::shared_ptr<AllocatedMemory> memory =
                std::make_shared<AllocatedMemory>(
                    output_state_r->Data()->TotalByteSize(), memory_type,
                    memory_type_id);
            RETURN_IF_ERROR(input_state_r->RemoveAllData());
            RETURN_IF_ERROR(input_state_r->SetData(output_state_r->Data()));
            RETURN_IF_ERROR(output_state_r->RemoveAllData());
            RETURN_IF_ERROR(output_state_r->SetData(memory));
          }
        }

        // Update the shape and data type of the output state if it doesn't
        // match the input state.
        if (input_state_r->Shape() != output_state_r->Shape()) {
          *input_state_r->MutableShape() = output_state_r->Shape();
        }

        if (input_state_r->DType() != output_state_r->DType()) {
          *input_state_r->MutableDType() = output_state_r->DType();
        }

        return Status::Success;
      });

  return Status::Success;
}

Status
SequenceStates::OutputState(
    const std::string& name, const inference::DataType datatype,
    const std::vector<int64_t>& shape, SequenceState** output_state)
{
  return OutputState(name, datatype, shape.data(), shape.size(), output_state);
}

std::shared_ptr<SequenceStates>
SequenceStates::CopyAsNull(const std::shared_ptr<SequenceStates>& from)
{
  std::shared_ptr<SequenceStates> lsequence_states;
  if (from != nullptr) {
    lsequence_states.reset(new SequenceStates);
    for (auto& from_input_state : from->InputStates()) {
      auto& from_input_state_tensor = from_input_state.second;
      const auto& input_pair = lsequence_states->input_states_.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(from_input_state_tensor->Name()),
          std::forward_as_tuple(new SequenceState(
              from_input_state_tensor->Name(), from_input_state_tensor->DType(),
              from_input_state_tensor->Shape(), false /* reused buffer */,
              false /* use growable memory */)));

      auto& input_tensor = input_pair.first->second;
      std::shared_ptr<AllocatedMemory> data;
      if (from_input_state_tensor->DType() ==
          inference::DataType::TYPE_STRING) {
        // Use all-zero input states for null requests.
        auto element_count =
            triton::common::GetElementCount(from_input_state_tensor->Shape());
        auto state_size = 4 * element_count;
        data = std::make_shared<AllocatedMemory>(
            state_size, TRITONSERVER_MEMORY_CPU, 0);
      } else {
        data = std::make_shared<AllocatedMemory>(
            from_input_state_tensor->Data()->TotalByteSize(),
            TRITONSERVER_MEMORY_CPU, 0);
      }

      input_tensor->SetData(data);
      if (input_tensor->DType() == inference::DataType::TYPE_STRING) {
        input_tensor->SetStringDataToZero();
      }
    }

    for (auto& from_output_state : from->OutputStates()) {
      lsequence_states->output_states_.emplace(
          std::piecewise_construct,
          std::forward_as_tuple(from_output_state.first),
          std::forward_as_tuple());
    }
  }
  return lsequence_states;
}
}}  // namespace triton::core
