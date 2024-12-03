#pragma once

#include <vulkan/vulkan.hpp>
#include <vma/vk_mem_alloc.h>

struct SurfaceData
{
	vk::DescriptorSet descriptorSet;

	vk::Buffer positionBuffer;
	VmaAllocation positionBufferMem;
	vk::Buffer normalBuffer;
	VmaAllocation normalBufferMem;
	vk::Buffer tangentBuffer;
	VmaAllocation tangentBufferMem;
	vk::Buffer albedoUVBuffer;
	VmaAllocation albedoUVBufferMem;
	vk::Buffer normalUVBuffer;
	VmaAllocation normalUVBufferMem;
	vk::Buffer metalRoughUVBuffer;
	VmaAllocation metalRoughUVBufferMem;

	vk::Buffer indexBuffer;
	VmaAllocation indexBufferMem;
	std::uint32_t indexNumber;

	vk::Image albedoImage;
	VmaAllocation albedoImageMem;
	vk::ImageView albedoImageView;
	vk::Image normalImage;
	VmaAllocation normalImageMem;
	vk::ImageView normalImageView;
	vk::Image metalRoughImage;
	VmaAllocation metalRoughImageMem;
	vk::ImageView metalRoughImageView;

	vk::PrimitiveTopology topology;
};