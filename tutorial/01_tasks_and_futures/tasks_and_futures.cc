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

/*
 * To illustrate task launches and futures in Legion
 * we implement a program to compute the first N
 * Fibonacci numbers.  While we note that this is not
 * the fastest way to compute Fibonacci numbers, it
 * is designed to showcase the functional nature of
 * Legion tasks and futures.
 */

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
  FIBONACCI_TASK_ID,
  SUM_TASK_ID,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_fibonacci = 7;
  printf("Computing the first %d Fibonacci numbers...\n", num_fibonacci);

  // Create a vector of Future results
  std::vector<Future> fib_results;

  // Compute the first num_fibonacci numbers
  for (int i = 0; i < num_fibonacci; i++)
  {
    // Create an instance of a task to launch, providing an argument
    TaskLauncher launcher(FIBONACCI_TASK_ID, TaskArgument(&i,sizeof(i)));
    // The execute_task method returns immediately and the task is scheduled
    fib_results.push_back(runtime->execute_task(ctx, launcher));
  }
  
  // Print out our results
  for (int i = 0; i < num_fibonacci; i++)
  {
    // The get_result method blocks until the task is complete
    int result = fib_results[i].get_result<int>(); 
    printf("Fibonacci(%d) = %d (elapsed = %.2f s)\n", i, result, elapsed);
  }
}

int fibonacci_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
  // The 'TaskArgument' value passed to a task and its size
  // in bytes is available in the 'args' and 'arglen' fields
  // on the 'Task' object.
  //
  // Since there is no type checking when writing to
  // the runtime API (a benefit provided by our Legion compiler)
  // we encourage programmers to check that they are getting
  // what they expect in their values.
  assert(task->arglen == sizeof(int));
  int fib_num = *(const int*)task->args; 
  // Fibonacci base cases
  // Note that tasks return values the same as C functions.
  // If a task is running remotely from its parent task then
  // Legion automatically packages up the result and returns
  // it to the origin location.
  if (fib_num == 0)
    return 0;
  if (fib_num == 1)
    return 1;

  // Launch fib-1
  const int fib1 = fib_num-1;
  TaskLauncher t1(FIBONACCI_TASK_ID, TaskArgument(&fib1,sizeof(fib1)));
  Future f1 = runtime->execute_task(ctx, t1);

  // Launch fib-2
  const int fib2 = fib_num-2;
  TaskLauncher t2(FIBONACCI_TASK_ID, TaskArgument(&fib2,sizeof(fib2)));
  Future f2 = runtime->execute_task(ctx, t2);

  // Here will illustrate a non-blocking way of using a future. 
  // Rather than waiting for the values and passing the results
  // directly to the summation task, we instead pass the futures
  // through the TaskLauncher object.  Legion then will 
  // ensure that the sum task does not begin until both futures
  // are ready and that the future values are available wherever
  // the sum task is run (even if it is run remotely).  Futures
  // should NEVER be passed through a TaskArgument.
  TaskLauncher sum(SUM_TASK_ID, TaskArgument(NULL, 0));
  sum.add_future(f1);
  sum.add_future(f2);
  Future result = runtime->execute_task(ctx, sum);

  // Our API does not permit returning Futures as the result of 
  // a task.  Any attempt to do so will result in a failed static 
  // assertion at compile-time.  In general, waiting for one or 
  // more futures at the end of a task is inexpensive since we 
  // have already exposed the available sub-tasks for execution 
  // to the Legion runtime so we can extract as much task-level
  // parallelism as possible from the application.
  return result.get_result<int>();
}

int sum_task(const Task *task,
             const std::vector<PhysicalRegion> &regions,
             Context ctx, Runtime *runtime)
{
  assert(task->futures.size() == 2);
  // Note that even though it looks like we are performing
  // blocking calls to get these future results, the
  // Legion runtime is smart enough to not run this task
  // until all the future values passed through the
  // task launcher have completed.
  Future f1 = task->futures[0];
  int r1 = f1.get_result<int>();
  Future f2 = task->futures[1];
  int r2 = f2.get_result<int>();

  return (r1 + r2);
}
              
int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  // Note that tasks which return values must pass the type of
  // the return argument as the first template parameter.

  {
    TaskVariantRegistrar registrar(FIBONACCI_TASK_ID, "fibonacci");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<int, fibonacci_task>(registrar, "fibonacci");
  }

  // The sum-task has a very special property which is that it is
  // guaranteed never to make any runtime calls.  We call these
  // kinds of tasks "leaf" tasks and tell the runtime system
  // about them using the 'TaskConfigOptions' struct.  Being
  // a leaf task allows the runtime to perform significant
  // optimizations that minimize the overhead of leaf task
  // execution.

  {
    TaskVariantRegistrar registrar(SUM_TASK_ID, "sum");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf(true);
    Runtime::preregister_task_variant<int, sum_task>(registrar, "sum");
  }

  return Runtime::start(argc, argv);
}
