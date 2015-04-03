#pragma once
//:
// \file
#include <vcl_string.h>
#include <vcl_iostream.h>
#include <vcl_vector.h>
#include <bocl/bocl_manager.h>
#include <bocl/bocl_utils.h>
#include <bocl/bocl_mem.h>
#include <bocl/bocl_kernel.h>

class bocl_poisson_manager : public bocl_manager<bocl_poisson_manager>
{
 public:

  bocl_poisson_manager()
    : program_(0),time_in_secs_(0.0f) {}

  ~bocl_poisson_manager();
  bool run_cpu(unsigned len);
  bool run_kernel();
  int create_kernel(vcl_string const& kernel_name, vcl_string src_path, vcl_string options);

  bocl_kernel kernel()      { return kernel_; }
  bool release_buffers();
  void create_buffers();
	//accessor
//  float getPhi(unsigned i){return phi_t_[i];}
  void setIter(unsigned l){iter_=l;}
  void setLen(unsigned l){len_=l;}
  void setdata(int len,int workgroupsize);
  void printtofile();
 protected:

  cl_program program_;
  
  cl_command_queue command_queue_;
  bocl_kernel kernel_;

  /*cl_float* array_;
  cl_float* result_array_;
  cl_uint * cl_len_;*/
  float * rho_;
  float * phi_t_;
  float * phi_tp1_;
  int * iternum_;
  //float * result_flag_;
  
  bocl_mem* rho_buf_;
  bocl_mem* phi_t_buf_;
  bocl_mem* phi_tp1_buf_;  
  bocl_mem* iter_buf_;



  float time_in_secs_;
  unsigned len_,iter_,workgroup_size_;
};


