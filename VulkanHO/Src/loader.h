#pragma once

#include <vulkan/vulkan.hpp>
#include <vma/vk_mem_alloc.h>
#include "sceneNode.h"
#include <tiny_gltf.h>
#include <type_traits>
#include <functional>

class Loader
{
public:
	struct LoaderArgs
	{
		vk::Device device;
		VmaAllocator allocator;
		vk::DescriptorPool descriptorPool;
		vk::DescriptorSetLayout descriptorSetLayout;
		vk::Sampler sampler;
		vk::Queue queue;
		vk::CommandBuffer commandBuffer;
		uint32_t queueFamilyIndex;
	};

	std::shared_ptr<SceneNode> loadGLTF(LoaderArgs const & args, std::string const & file);

private:
	enum class TextureType : uint8_t
	{
		NotSet = 0b0,
		Albedo = 0b1,
		Normal = 0b10,
		MetallicRoughness = 0b100,
		Emissive = 0b1000,
		Occlusion = 0b10000
	};

	friend constexpr Loader::TextureType operator|(const Loader::TextureType t1, const Loader::TextureType t2)
	{
		return
			static_cast<Loader::TextureType>
			(
				static_cast<std::underlying_type_t<Loader::TextureType>>(t1) |
				static_cast<std::underlying_type_t<Loader::TextureType>>(t2)
			);
	};

	friend constexpr Loader::TextureType operator&(const Loader::TextureType t1, const Loader::TextureType t2)
	{
		return
			static_cast<Loader::TextureType>
			(
				static_cast<std::underlying_type_t<Loader::TextureType>>(t1) &
				static_cast<std::underlying_type_t<Loader::TextureType>>(t2)
			);
	};

	friend constexpr Loader::TextureType& operator|=(Loader::TextureType& t1, const Loader::TextureType t2)
	{
		return t1 =
			static_cast<Loader::TextureType>
			(
				static_cast<std::underlying_type_t<Loader::TextureType>>(t1) |
				static_cast<std::underlying_type_t<Loader::TextureType>>(t2)
			);
	};

	friend constexpr Loader::TextureType operator~(const Loader::TextureType t1)
	{
		return static_cast<Loader::TextureType>(~static_cast<std::underlying_type_t<Loader::TextureType>>(t1));
	};
	
	tinygltf::Model model;
	vk::Device device;
	VmaAllocator allocator;
	vk::DescriptorPool descriptorPool;
	vk::DescriptorSetLayout descriptorSetLayout;
	vk::Sampler sampler;
	vk::Queue queue;
	vk::CommandBuffer commandBuffer;
	uint32_t queueFamilyIndex;

	std::vector<std::pair<vk::Buffer, VmaAllocation>> buffers;
	std::vector<vk::DeviceSize> bufferSizes;
	std::vector<std::pair<vk::Buffer, VmaAllocation>> stagingBuffers;
	std::vector<std::pair<vk::Image, VmaAllocation>> images;
	std::vector<vk::Extent3D> imageExtents;
	std::vector<std::pair<vk::Buffer, VmaAllocation>> stagingImageBuffers;
	std::vector<vk::Sampler> imageSamplers;
	std::vector<vk::ImageView> imageViews;

	void unpackArgs(LoaderArgs const & args);

	void loadGLTFHelper(std::string const & file);

	void createBuffers();
	void createImages();
	vk::Filter getMagnificationFilter(tinygltf::Sampler const& sampler);
	vk::Filter getMinificationFilter(tinygltf::Sampler const& sampler, vk::SamplerMipmapMode & mipMapMode);
	void getSamplerAddressModes(tinygltf::Sampler const& sampler, vk::SamplerAddressMode & sAddressMode, vk::SamplerAddressMode & tAddressMode);
	void createNodes(std::shared_ptr<SceneNode>& sceneNode, tinygltf::Node const & node);

	vk::Format getImageFormat(tinygltf::Image const & image);
	vk::Format getSRGBImageFormat(tinygltf::Image const & image);
	
	void getNodeLocalMatrix(tinygltf::Node const & node, std::shared_ptr<SceneNode> const & sceneNode);
	size_t getBytePerComponent(size_t compTypeID);
	size_t getNumberOfComponents(size_t typeID);
	size_t getAccessorElementSize(size_t compTypeID, size_t typeID);

	void setupCommandBuffer();
	void recordCommandBuffer();
	void submitCommandBuffer();

	void freeStagingMemory();
};