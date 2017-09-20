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
 * In this section we use a sequential
 * implementation of daxpy to show how
 * to create physical instances of logical
 * regions.  In later sections we will
 * show how to extend this daxpy example
 * so that it will run with sub-tasks
 * and also run in parallel.
 */

// Note since we are now accessing data inside
// of logical regions we need the accessor namespace.
using namespace LegionRuntime::Accessor;

enum TaskIDs {
  TOP_LEVEL_TASK_ID,
};

enum FieldIDs {
  FID_X,
  FID_Y,
  FID_Z,
};

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  int num_elements = 1024; 
  // See if we have any command line arguments to parse
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for (int i = 1; i < command_args.argc; i++)
    {
      if (!strcmp(command_args.argv[i],"-n"))
        num_elements = atoi(command_args.argv[++i]);
    }
  }
  printf("Running daxpy for %d elements...\n", num_elements);

  Rect<1> elem_rect(Point<1>(0),Point<1>(num_elements-1));
  IndexSpace is = runtime->create_index_space(ctx, 
                          Domain::from_rect<1>(elem_rect));
  FieldSpace input_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, input_fs);
    allocator.allocate_field(sizeof(double),FID_X);
    allocator.allocate_field(sizeof(double),FID_Y);
  }
  FieldSpace output_fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = 
      runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(double),FID_Z);
  }

  // LogicalRegions created from IndexSpace and FieldSpace
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, input_fs);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, output_fs);

  // Access requirements into logical region
  RegionRequirement req(input_lr, READ_WRITE, EXCLUSIVE, input_lr);
  req.add_field(FID_X);
  req.add_field(FID_Y);
  InlineLauncher input_launcher(req);

  // Obtain access to the the physical region
  PhysicalRegion input_region = runtime->map_region(ctx, input_launcher);
  input_region.wait_until_valid();

  // Obtain accessors into fields of the physical region
  RegionAccessor<AccessorType::Generic, double> acc_x = 
    input_region.get_field_accessor(FID_X).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> acc_y = 
    input_region.get_field_accessor(FID_Y).typeify<double>();

  // Iterate through the index of the physical region
  for (GenericPointInRectIterator<1> pir(elem_rect); pir; pir++)
  {
    acc_x.write(DomainPoint::from_point<1>(pir.p), drand48());
    acc_y.write(DomainPoint::from_point<1>(pir.p), drand48());
  }

  InlineLauncher output_launcher(RegionRequirement(output_lr, WRITE_DISCARD,
                                                   EXCLUSIVE, output_lr));
  output_launcher.requirement.add_field(FID_Z);

  // Map the region
  PhysicalRegion output_region = runtime->map_region(ctx, output_launcher);

  // This accessor invokes the implicit 'wait_until_valid' call.
  RegionAccessor<AccessorType::Generic, double> acc_z = 
    output_region.get_field_accessor(FID_Z).typeify<double>();

  const double alpha = drand48();
  for (GenericPointInRectIterator<1> pir(elem_rect); pir; pir++)
  {
    double value = alpha * acc_x.read(DomainPoint::from_point<1>(pir.p)) + 
                           acc_y.read(DomainPoint::from_point<1>(pir.p));
    acc_z.write(DomainPoint::from_point<1>(pir.p), value);
  }
  printf("Done!\n");

  // In some cases it may be necessary to unmap regions and then 
  // remap them.  We'll give a compelling example of this in the
  // next example.   In this case we'll remap the output region
  // with READ-ONLY privileges to check the output result.
  // We really could have done this directly since WRITE-DISCARD
  // privileges are equivalent to READ-WRITE privileges in terms
  // of allowing reads and writes, but we'll explicitly unmap
  // and then remap.  Unmapping is done with the unmap call.
  // After this call the physical region no longer contains valid
  // data and all accessors from the physical region are invalidated.
  runtime->unmap_region(ctx, output_region);

  // We can then remap the region.  Note if we wanted to remap
  // with the same privileges we could have used the 'remap_region'
  // call.  However, we want different privileges so we update
  // the launcher and then remap the region.  The 'remap_region' 
  // call also guarantees that we would get the same physical 
  // instance.  By calling 'map_region' again, we have no such 
  // guarantee.  We may get the same physical instance or a new 
  // one.  The orthogonality of correctness from mapping decisions
  // ensures that we will access the same data regardless.
  output_launcher.requirement.privilege = READ_ONLY;
  output_region = runtime->map_region(ctx, output_launcher);

  // Since we may have received a new physical instance we need
  // to update our accessor as well.  Again this implicitly calls
  // 'wait_until_valid' to ensure we have valid data.
  acc_z = output_region.get_field_accessor(FID_Z).typeify<double>();

  printf("Checking results...");
  bool all_passed = true;
  // Check our results are the same
  for (GenericPointInRectIterator<1> pir(elem_rect); pir; pir++)
  {
    double expected = alpha * acc_x.read(DomainPoint::from_point<1>(pir.p)) + 
                           acc_y.read(DomainPoint::from_point<1>(pir.p));
    double received = acc_z.read(DomainPoint::from_point<1>(pir.p));
    // Probably shouldn't check for floating point equivalence but
    // the order of operations are the same should they should
    // be bitwise equal.
    if (expected != received)
      all_passed = false;
  }
  if (all_passed)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");

  // Clean up all our data structures.
  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, input_fs);
  runtime->destroy_field_space(ctx, output_fs);
  runtime->destroy_index_space(ctx, is);
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
