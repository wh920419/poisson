#include <testlib/testlib_test.h>
#include <testlib/testlib_root_dir.h>
#include "bocl_poisson_manager.h" 
#include "vcl_fstream.h"
#include "vul/vul_timer.h"
#include <iostream>
#include <vector>
using namespace std;
bool run_poisson(unsigned len,unsigned workgroup_size,unsigned iterations)
{
		
	//instantiate your manager here
	bocl_poisson_manager * mgr=bocl_poisson_manager::instance();
	mgr->setIter(iterations);
	mgr->run_cpu(len);//run cpu
	
	//use it to create the kernel; pass any relevant arguments
	/*
	vul_timer t1;
	mgr->setdata(len,workgroup_size);
	string root_dir = testlib_root_dir();//vcl_
	mgr->create_kernel("poisson",root_dir + "/contrib/brl/bbas/bocl/tests/poisson.cl", "");
	  if (!mgr->create_kernel("poisson",root_dir +
    "/contrib/brl/bbas/bocl/tests/poisson.cl", "")) {	
		TEST("Create Kernel test_poisson", false, true);
		return false;
	}
	//run the kernel
	if (mgr->run_kernel()!=SDK_SUCCESS) {
		TEST("Run Kernel test_poisson", false, true);
		return false;
	}
	//output the results in a file out.txt and print the total runtime
	mgr->printtofile();
	cout<<"********use**********"<<endl;
	cout<<t1.real()<<endl;
	cout<<"msec to run**** "<<endl;
	*/

  return true;
}
static void test_poisson(){
	//unsigned len=100000;
	unsigned len=10000;
	unsigned gsize=16;
	unsigned iterations=150000;
	run_poisson(len,gsize,iterations);
}
TESTMAIN(test_poisson);