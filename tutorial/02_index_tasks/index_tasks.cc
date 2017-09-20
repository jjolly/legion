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
 * This example is a redux version of hello world 
 * which shows how launch a large array of tasks
 * using a single runtime call.  We also describe
 * the basic Legion types for arrays, domains,
 * and points and give examples of how they work.
 */

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  INDEX_SPACE_TASK_ID,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_points = 4;
  printf("Running hello world redux for %d points...\n", num_points);

  // Index space bounds are represented by a Rect object and
  // is defined by Point object, inclusively
  Rect<1> launch_bounds(Point<1>(0),Point<1>(num_points-1));
  // Domain object is a repreentation of a specific range
  Domain launch_domain = Domain::from_rect<1>(launch_bounds);

  // An ArgumentMap defines the inputs for a set of tasks
  ArgumentMap arg_map;
  for (int i = 0; i < num_points; i++)
  {
    int input = i + 10;
    // Domain objects are generic representations of specific dimensions
    arg_map.set_point(DomainPoint::from_point<1>(Point<1>(i)),
        TaskArgument(&input,sizeof(input)));
  }
  // IndexLauncher prepares several targets to be launches
  IndexLauncher index_launcher(INDEX_SPACE_TASK_ID,
                               launch_domain,
                               TaskArgument(NULL, 0),
                               arg_map);
  // The FutureMap is a collection of results from tasks
  FutureMap fm = runtime->execute_index_space(ctx, index_launcher);

  // Now the top-level tasks waits for the sub-tasks
  fm.wait_all_results();

  for (int i = 0; i < num_points; i++)
  {
    int expected = 2*(i+10);
    // And gets the results from each sub-task
    int received = fm.get_result<int>(DomainPoint::from_point<1>(Point<1>(i)));
    printf("Task #%d: Expected: %d, Received: %d\n", i, expected, received);
  }
}

int index_space_task(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  // The point for this task is available in the task
  // structure under the 'index_point' field.
  assert(task->index_point.get_dim() == 1); 
  printf("Hello world from task %lld!\n", task->index_point.point_data[0]);
  // Values passed through an argument map are available 
  // through the local_args and local_arglen fields.
  assert(task->local_arglen == sizeof(int));
  int input = *((const int*)task->local_args);
  return (2*input);
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  {
    TaskVariantRegistrar registrar(INDEX_SPACE_TASK_ID, "index_space_task");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<int, index_space_task>(registrar, "index_space_task");
  }

  return Runtime::start(argc, argv);
}
