__kernel void poisson(__global float* rho,
                      __global float* phi_t,
                      __global float* phi_tp1,
					  __global int* iter,
					  //__global int* result_flag,
					  __local float* loc_rho,
                      __local float* loc_phi_t,
                      __local float* loc_phi_tp1){ 

    unsigned int tid = get_local_id(0);
    unsigned int gid = get_global_id(0);
	unsigned int bid = get_group_id(0);
	unsigned int localsize = get_local_size(0); 
    unsigned int groupnum = get_num_groups(0);
	unsigned int globalsize = get_global_size(0);

	int iternum = iter[0];
		int index=gid-bid*2 ;//blocksize * blockid + localid-(overlap offset)bid*localsize+lid-bid

	
	for(unsigned i=0;i<iternum;i++){
		loc_rho[tid]=rho[gid];
		loc_phi_t[tid]=phi_t[gid];
		loc_phi_tp1[tid]=phi_tp1[gid];	

		barrier(CLK_LOCAL_MEM_FENCE);
	
		if(tid==0){
			if(bid==0){
				loc_phi_tp1[tid]=(0+loc_phi_t[tid+1]-loc_rho[tid])/2;
			}else{
				loc_phi_tp1[tid]=(loc_phi_t[tid+1]+phi_t[gid-1]-loc_rho[tid])/2;			
			}				
		}
		else if(tid==localsize-1){
			if(bid==groupnum-1){
				loc_phi_tp1[tid]=(0+loc_phi_t[tid-1]-loc_rho[tid])/2;
			}else{
				loc_phi_tp1[tid]=(phi_t[gid+1]+loc_phi_t[tid-1]-loc_rho[tid])/2;			
			}
		}
		else{
			loc_phi_tp1[tid]=(loc_phi_t[tid+1]+loc_phi_t[tid-1]-loc_rho[tid])/2;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		phi_t[gid]=loc_phi_tp1[tid];
		phi_tp1[gid]=loc_phi_t[tid];
		barrier(CLK_LOCAL_MEM_FENCE);

	}	


/*
	if(bid==groupnum-1)

	for(unsigned i=0;i<1000;i++){
		loc_phi_t[lid]=(loc_phi_tp1[lid-1]+loc_phi_tp1[lid+1]-loc_rho[lid])/2;		
		loc_phi_tp1=loc_phi_t;
	}
*/	
  return;
}

