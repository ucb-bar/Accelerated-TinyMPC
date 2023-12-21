#ifndef __ROSE_H__
#define __ROSE_H__

#include <stdint.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "mmio.h"

#define ROSE_PHYS_ADDR 0x2000

#define ROSE_STATUS (rose_ptr + 0x00)
#define ROSE_IN     (rose_ptr + 0x08)
#define ROSE_OUT    (rose_ptr + 0x0C)
#define ROSE_WRITTEN_COUNTER_MAX (rose_ptr + 0x14)

//TODO: verify this is the correct address and cacheblockbytes
#define ROSE_DMA 0x88000000
#define CacheBlockBytes 64

#define ROSE_RESET 0x01

intptr_t rose_ptr = 0x2000;

void map_rose_ptr() {
	int mem_fd;
	mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
	rose_ptr = (intptr_t) mmap(NULL, 16, PROT_READ 
								| PROT_WRITE , 
								MAP_SHARED, 
								mem_fd, 
								ROSE_PHYS_ADDR);
	printf("RoSE ptr virtual addr: %x\n", rose_ptr);
}

void send_sim_reset() {
	// printf("Sending reset cmd...\n");
    while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
    reg_write32(ROSE_IN, ROSE_RESET);
    while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
    reg_write32(ROSE_IN, 0);
}

void send_obs_req(uint8_t cmd) {
    // printf("Requesting cmd %02x...\n", cmd);
    while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
    reg_write32(ROSE_IN, cmd);
    while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
    reg_write32(ROSE_IN, 0);
}

void read_obs_rsp(void * buf) {
	uint32_t i;
	uint8_t status;
	uint32_t raw_result;
	float result;

	uint32_t * raw_buf = (uint32_t*) buf;
	// printf("Receiving obs...\n");
	while ((reg_read8(ROSE_STATUS) & 0x4) == 0) ;
	uint32_t cmd = reg_read32(ROSE_OUT);
	// printf("Got cmd: %x\n", cmd);
	while ((reg_read8(ROSE_STATUS) & 0x4) == 0) ;
	uint32_t num_bytes = reg_read32(ROSE_OUT);
	// printf("Got num bytes: %d\n", num_bytes);
	for(i = 0; i < num_bytes / 4; i++) {
		while ((reg_read8(ROSE_STATUS) & 0x4) == 0) ;
    	raw_buf[i] = reg_read32(ROSE_OUT);
    	// printf("(%d, %x) ", i, buf[i]);
	}
	
}

int read_obs_rsp_nonblock(void * buf) {
	uint32_t i;
	uint8_t status;
	uint32_t raw_result;
	float result;

	uint32_t * raw_buf = (uint32_t*) buf;
	// printf("Receiving obs...\n");
	if ((reg_read8(ROSE_STATUS) & 0x4) == 0) {
		return 0;
	}
	uint32_t cmd = reg_read32(ROSE_OUT);
	// printf("Got cmd: %x\n", cmd);
	while ((reg_read8(ROSE_STATUS) & 0x4) == 0) ;
	uint32_t num_bytes = reg_read32(ROSE_OUT);
	// printf("Got num bytes: %d\n", num_bytes);
	for(i = 0; i < num_bytes / 4; i++) {
		while ((reg_read8(ROSE_STATUS) & 0x4) == 0) ;
    	raw_buf[i] = reg_read32(ROSE_OUT);
    	// printf("(%d, %x) ", i, buf[i]);
	}
	return 1;
	
}


void send_action(void * buf, uint32_t cmd, uint32_t num_bytes) {
	uint32_t i;
	uint32_t * raw_buf = (uint32_t*) buf;

	// printf("Sending action %x...\n", cmd);
    while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
    reg_write32(ROSE_IN, cmd);
    while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
    reg_write32(ROSE_IN, num_bytes);
	for (i = 0; i < num_bytes / 4; i++) {
		while ((reg_read8(ROSE_STATUS) & 0x1) == 0) ;
		reg_write32(ROSE_IN, raw_buf[i]);
	}

}

#endif
