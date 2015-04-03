#include "bocl_poisson_manager.h"
#include <vcl_cstdio.h>
#include <ctime>
#include <vul/vul_timer.h>
#include <vector>
using namespace std;
#define local_workgroup_size 32
//#define printout 0
bocl_poisson_manager::~bocl_poisson_manager(){}

void bocl_poisson_manager::create_buffers()
{  
	//cl_int * result_flag_;
 // bocl_mem* phi_t_;
 // bocl_mem* phi_tp1_;
 // bocl_mem* result_flag_buf_;

  rho_buf_ = new bocl_mem(this->context(), rho_, len_ * sizeof(cl_float), "rho_buf_");
  rho_buf_->create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

  phi_t_buf_ = new bocl_mem(this->context(), phi_t_, len_ * sizeof(cl_float), "phi_t_buf_");
  phi_t_buf_->create_buffer(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);

  phi_tp1_buf_ = new bocl_mem(this->context(), phi_tp1_, len_ * sizeof(cl_float), "phi_tp1_buf_");
  phi_tp1_buf_->create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);

  iter_buf_ = new bocl_mem(this->context(), iternum_, sizeof(cl_int), "iter_buf_ ");
  iter_buf_ ->create_buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
}

bool bocl_poisson_manager::run_kernel()
{
  cl_int status = CL_SUCCESS;

  // --Create and initialize memory objects here--
  create_buffers();
  // -- Set appropriate arguments to the kernel  here--
  kernel_.set_arg( rho_buf_ );//rho
  kernel_.set_arg( phi_t_buf_ );//phi1
  kernel_.set_arg( phi_tp1_buf_ );//phi2
  kernel_.set_arg( iter_buf_ );//iteration
  //local argument
  kernel_.set_local_arg(sizeof(cl_float)*this->group_size());
  kernel_.set_local_arg(sizeof(cl_float)*this->group_size());
  kernel_.set_local_arg(sizeof(cl_float)*this->group_size());

  cl_ulong used_local_memory;
  status = clGetKernelWorkGroupInfo(kernel_.kernel(),this->devices()[0],CL_KERNEL_LOCAL_MEM_SIZE,
                                    sizeof(cl_ulong),&used_local_memory,NULL);
  if (!check_val(status,CL_SUCCESS,"clGetKernelWorkGroupInfo CL_KERNEL_LOCAL_MEM_SIZE failed."))
    return SDK_FAILURE;

  // determine the work group size
  cl_ulong kernel_work_group_size;
  status = clGetKernelWorkGroupInfo(kernel_.kernel(),this->devices()[0],CL_KERNEL_WORK_GROUP_SIZE,
                                    sizeof(cl_ulong),&kernel_work_group_size,NULL);
  if (!check_val(status,CL_SUCCESS,"clGetKernelWorkGroupInfo CL_KERNEL_WORK_GROUP_SIZE, failed."))
    return SDK_FAILURE;

  vcl_size_t globalThreads[]= {RoundUp(len_,workgroup_size_)};
  vcl_size_t localThreads[] = {workgroup_size_};

  if (used_local_memory > this->total_local_memory())
  {
    vcl_cout << "Unsupported: Insufficient local memory on device.\n";
    return SDK_FAILURE;
  }

  // set up a command queue
  command_queue_ = clCreateCommandQueue(this->context(),this->devices()[0],CL_QUEUE_PROFILING_ENABLE,&status);
  if (!check_val(status,CL_SUCCESS,"Falied in command queue creation" + error_to_string(status)))
	  return false;
 
  //execute the kernel based on the number of iterations 
  kernel_.execute(command_queue_, 1, localThreads, globalThreads);
  

  status = clFinish(command_queue_); //this line should go outside your iteration loop.
    if (!check_val(status,CL_SUCCESS,"clFinish failed."+error_to_string(status)))
	  return SDK_FAILURE;  
 //read buffer
	rho_buf_->read_to_buffer(command_queue_);
	phi_t_buf_->read_to_buffer(command_queue_);
    phi_tp1_buf_->read_to_buffer(command_queue_);



	status = clReleaseCommandQueue(command_queue_);
  if (!check_val(status,CL_SUCCESS,"clReleaseCommandQueue failed."))
    return SDK_FAILURE;
  release_buffers();
  return CL_SUCCESS;
  
}

bool bocl_poisson_manager::release_buffers(){
	//free up your cl_mem objects
  rho_buf_->release_memory();
  phi_t_buf_->release_memory();
  phi_tp1_buf_->release_memory();
  kernel_.clear_args();
	return true;
}

int bocl_poisson_manager::create_kernel(vcl_string const& kernel_name,
                                                        vcl_string src_path,
                                                        vcl_string options)
{
  vcl_vector<vcl_string> src_paths;
  src_paths.push_back(src_path);
  return kernel_.create_kernel(&this->context(),&this->devices()[0],
                               src_paths, kernel_name, options, "the kernel");

}


void bocl_poisson_manager::setdata(int len,int w){
	this->len_=len;
	int *ii=new int[1];	ii[0]=this->iter_; this->iternum_=ii;
	//int ii[1]={this->iter_}; this->iternum_=ii;//se iteration
	this->workgroup_size_=w;
	//allocate varialbles on CPU
	rho_=new float[len];
	phi_t_=new float[len];
	phi_tp1_=new float[len];
	for(int i=0;i<len;i++){
		rho_[i]=0.0;
		phi_t_[i]=0.0;
		phi_tp1_[i]=0.0;		
	}
	//a)
	//rho_[ (int)(len*(1.0/3.0)) ]=30;
	//rho_[ (int)(len*(2.0/3.0)) ]=-30;	

	//another way
	for(unsigned i=0;i<len;i++){
		int a=i%4;
		if( (i%4)==0||(i%4)==1 ){
			rho_[i]=-30;
		}else if((i%4)==2||(i%4)==3 ){
			rho_[i]=20;
		}
	}

}

void bocl_poisson_manager::printtofile(){
	vcl_ofstream ofs("out_gpua.txt");
	ofs<<"***************GPU: possion a)*********"<<endl;
	for(unsigned j=0;j<len_;j++){
		//float a=phi_t_[j];
		ofs<<phi_t_[j]<<endl;
		//ofs<<phi_tp1_[j]<<endl;
	}
	
}

bool bocl_poisson_manager::run_cpu(unsigned len){
	unsigned iterations=this->iter_;
	vul_timer t1;
	//A) 0 everywhere besides two values found at indexes 1/3* (length) in and (2/3) *length respectively. 
	//The first should be 30 and the second should be -30.
	vcl_ofstream ofs("out_cpu.txt");
	//implement the  run_cpu method with the poisson solver and call it here; report the total run time
	vector<float> input(len,0.0);	
	input[(int)len*(1.0/3.0)]=30;
	input[(int)len*(2.0/3.0)]=-30;
	vector<float> phi(len+1,0.0);
	vector<float> phi2(len+1,0.0);
	for(unsigned j=0;j<iterations;j++){
		for(unsigned i=1;i<len;i++){
			phi[i]=(phi2[i-1]+phi2[i+1]-input[i])/2;
		}
		std::swap(phi,phi2);		
	}
	ofs<<"***************CPU: possion a)*********"<<endl;
	for(unsigned j=0;j<len+1;j++){
		ofs<<phi[j]<<endl;
	}
	ofs<<t1.real()<<endl;
	ofs.close();
/*
	//B) The sequence [20 -30 -30 20] repeated until the whole array is filled.
	vul_timer t2;
	vcl_ofstream ofs2("out_cpu2.txt");	
	//implement the  run_cpu method with the poisson solver and call it here; report the total run time
	vector<float> input2(len,0.0);	
	for(unsigned i=0;i<len;i++){
		int a=i%4;
		if( (i%4)==0||(i%4)==1 ){
			input2[i]=-30;
		}else if((i%4)==2||(i%4)==3 ){
			input2[i]=20;
		}
	}
	phi.assign(len+1,0.0);
	phi2.assign(len+1,0.0);
	for(unsigned j=0;j<iterations;j++){
		for(unsigned i=1;i<len;i++){
			phi[i]=(phi2[i-1]+phi2[i+1]-input2[i])/2;
		}
		std::swap(phi,phi2);		
	}
	ofs2<<"***************CPU: possion b)*********"<<endl;
	for(unsigned j=0;j<len+1;j++){
		ofs2<<phi[j]<<endl;
	}
	ofs<<t2.real()<<endl;
	ofs2.close();	
	*/
	return true;
}