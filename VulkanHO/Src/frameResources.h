#pragma once

#include <vulkan/vulkan.hpp>
#include <vma/vk_mem_alloc.h>

struct FrameResources
{
	vk::Image posImage;
	vk::ImageView posImageView;
	VmaAllocation posImageMem;
	vk::Image albedoImage;
	vk::ImageView albedoImageView;
	VmaAllocation albedoImageMem;
	vk::Image normalImage;
	vk::ImageView normalImageView;
	VmaAllocation normalImageMem;
	vk::Image metalRoughImage;
	vk::ImageView metalRoughImageView;
	VmaAllocation metalRoughImageMem;
	vk::Image depthImage;
	vk::ImageView depthImageView;
	VmaAllocation depthImageMem;
	vk::CommandBuffer commBuff;
	vk::Semaphore swapchainSem;
	vk::Semaphore renderSem;
	vk::Fence renderFence;
};