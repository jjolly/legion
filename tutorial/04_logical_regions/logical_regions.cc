/* Copyright 2017 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include "legion.h"
using namespace Legion;

// for Point<DIM> and Rect<DIM>
using namespace LegionRuntime::Arrays;

/*
 * This example shows how to create index
 * spaces, field spaces, and logical regions.
 * It also shows how to dynamically allocate
 * and free elements in index spaces and fields
 * in field spaces.
 */

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
};

enum FieldIDs {
  FID_FIELD_A,
  FID_FIELD_B,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  // Unstructured IndexSpace has no fields allocated, but has an upper bound
  IndexSpace unstructured_is = runtime->create_index_space(ctx, 1024);
  // Structured IndexSpace has a specific number of fields allocated
  Rect<1> rect(Point<1>(0),Point<1>(1023));
  IndexSpace structured_is = runtime->create_index_space(ctx, 
                                          Domain::from_rect<1>(rect));
  // FieldSpace defines columns of information to be used by IndexSpace
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    FieldID fida = allocator.allocate_field(sizeof(double), FID_FIELD_A);
    assert(fida == FID_FIELD_A);
    FieldID fidb = allocator.allocate_field(sizeof(int), FID_FIELD_B);
    assert(fidb == FID_FIELD_B);
  }

  // LogicalRegion combines IndexSpace and FieldSpace to create data
  LogicalRegion unstructured_lr = 
    runtime->create_logical_region(ctx, unstructured_is, fs);
  LogicalRegion structured_lr = 
    runtime->create_logical_region(ctx, structured_is, fs);

  // Each LogicalRegion is different
  LogicalRegion no_clone_lr =
    runtime->create_logical_region(ctx, structured_is, fs);
  assert(structured_lr.get_tree_id() != no_clone_lr.get_tree_id());

  runtime->destroy_logical_region(ctx, unstructured_lr);
  runtime->destroy_logical_region(ctx, structured_lr);
  runtime->destroy_logical_region(ctx, no_clone_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, unstructured_is);
  runtime->destroy_index_space(ctx, structured_is);
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  return Runtime::start(argc, argv);
}

