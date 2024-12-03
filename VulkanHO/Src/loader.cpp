#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_USE_CPP14
#define STB_IMAGE_IMPLEMENTATION
#include "loader.h"
#include <iostream>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include "utils.hpp"

std::shared_ptr<SceneNode> Loader::loadGLTF(LoaderArgs const & args, std::string const & file)
{
	loadGLTFHelper(file);
	unpackArgs(args);

	buffers.resize(model.accessors.size());
	stagingBuffers.resize(model.accessors.size());
	bufferSizes.resize(model.accessors.size());
	createBuffers();
	
	images.resize(model.textures.size());
	stagingImageBuffers.resize(model.textures.size());
	imageExtents.resize(model.textures.size());
	imageSamplers.resize(model.textures.size());
	imageViews.resize(model.textures.size());
	createImages();
	
	setupCommandBuffer();
	recordCommandBuffer();
	submitCommandBuffer();
	
	std::shared_ptr<SceneNode> rootSceneNode = std::make_shared<SceneNode>();
	size_t defaultScene = (model.defaultScene > -1) ? model.defaultScene : 0;
	createNodes(rootSceneNode, model.nodes[model.scenes[defaultScene].nodes[0]]);
	
	device.waitIdle();
	freeStagingMemory();
	
	return rootSceneNode;
}

void Loader::createBuffers()
{
	for (size_t accessorIdx = 0; tinygltf::Accessor const& accessor : model.accessors)
	{
		size_t count = accessor.count;
		size_t elementSize = getAccessorElementSize(accessor.componentType, accessor.type);
		vk::DeviceSize bufferSize = count * elementSize;
		vk::BufferCreateInfo stagingBufferCreateInfo
		{
			{},
			bufferSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::SharingMode::eExclusive,
			1, &queueFamilyIndex
		};
		VmaAllocationCreateInfo stagingBufferVmaAllocInfo
		{
			.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO
		};
		vk::Buffer stagingBuffer;
		VmaAllocation stagingBufferMem;
		VmaAllocationInfo stagingBufferMemInfo;
		vmaCreateBuffer
		(
			allocator,
			reinterpret_cast<VkBufferCreateInfo*>(&stagingBufferCreateInfo), &stagingBufferVmaAllocInfo,
			reinterpret_cast<VkBuffer*>(&stagingBuffer), &stagingBufferMem, &stagingBufferMemInfo
		);
		stagingBuffers[accessorIdx] = std::make_pair(stagingBuffer, stagingBufferMem);


		tinygltf::Accessor::Sparse const& sparse = accessor.sparse;
		if (sparse.isSparse)
		{
			size_t replaceCount = sparse.count;
			std::vector<uint32_t> replaceIndices(replaceCount);
			tinygltf::BufferView const& replaceIndicesBufferView = model.bufferViews[sparse.indices.bufferView];
			size_t replaceIndicesOffset = sparse.indices.byteOffset + replaceIndicesBufferView.byteOffset;
			size_t replaceIndexSize = getBytePerComponent(sparse.indices.componentType);
			tinygltf::Buffer const& replaceIndicesBuffer = model.buffers[replaceIndicesBufferView.buffer];
			if (replaceIndexSize == 1)
			{ // Here I am assuming the machine is little endian, since gltf stores data that way, this could be trouble
				for (size_t i = 0; i < replaceCount; i++)
				{
					replaceIndices[i] = replaceIndicesBuffer.data[replaceIndicesOffset + i];
				}
			}
			else if (replaceIndexSize == 2)
			{
				for (size_t i = 0; i < replaceCount; i++)
				{
					uint16_t const& ind = replaceIndicesBuffer.data[replaceIndicesOffset + 2 * i];
					replaceIndices[i] = ind;
				}
			}
			else if (replaceIndexSize == 4)
			{
				for (size_t i = 0; i < replaceCount; i++)
				{
					uint32_t const& ind = replaceIndicesBuffer.data[replaceIndicesOffset + 4 * i];
					replaceIndices[i] = ind;
				}
			}


			tinygltf::BufferView const& replaceValuesBufferView = model.bufferViews[sparse.values.bufferView];
			size_t replaceValuesOffset = sparse.values.byteOffset + replaceValuesBufferView.byteOffset;
			size_t replaceValuesStride = replaceValuesBufferView.byteStride;
			if (replaceValuesStride == 0) replaceValuesStride = elementSize;
			tinygltf::Buffer const& replaceValuesBuffer = model.buffers[replaceValuesBufferView.buffer];
			tinygltf::BufferView const& bufferView = model.bufferViews[accessor.bufferView];
			size_t offset = accessor.byteOffset + bufferView.byteOffset;
			size_t stride = bufferView.byteStride;
			if (stride == 0) stride = elementSize;
			tinygltf::Buffer const& buffer = model.buffers[bufferView.buffer];
			uint8_t* stagingBufferData = static_cast<uint8_t*>(stagingBufferMemInfo.pMappedData);
			bool checkForReplace = replaceCount > 0;
			for (size_t i = 0, j = 0; i < count; i++)
			{ // Assume buffer contents not to be changed
				if (checkForReplace && replaceIndices[j] == i)
				{ // Since indices are stored in a strictly increasing order it suffices to have an increasing j variable to index them
					j++;
					checkForReplace = j < replaceCount;
					memcpy(stagingBufferData + elementSize * i, replaceValuesBuffer.data.data() + replaceValuesOffset + replaceValuesStride * i, elementSize);
				}
				else
				{
					memcpy(stagingBufferData + elementSize * i, buffer.data.data() + offset + stride * i, elementSize);
				}
			}
		}
		else
		{
			tinygltf::BufferView const& bufferView = model.bufferViews[accessor.bufferView];
			size_t offset = accessor.byteOffset + bufferView.byteOffset;
			size_t stride = bufferView.byteStride;
			tinygltf::Buffer const& buffer = model.buffers[bufferView.buffer];
			uint8_t* stagingBufferData = static_cast<uint8_t*>(stagingBufferMemInfo.pMappedData);
			if (stride == 0)
			{
				memcpy(stagingBufferData, buffer.data.data() + offset, elementSize * count); // substitute elem*count with buffsize
			}
			else
			{
				for (size_t i = 0, j = 0; i < count; i++) // remove j
				{ // Assume buffer contents not to be changed
					memcpy(stagingBufferData + elementSize * i, buffer.data.data() + offset + stride * i, elementSize);
				}
			}
		}


		vk::BufferCreateInfo destBufferCreateInfo
		{
			{},
			bufferSize,
			vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer, // Assume all buffers to be used in the same way
			vk::SharingMode::eExclusive,
			1,
			&queueFamilyIndex
		};
		VmaAllocationCreateInfo destBufferVmaAllocInfo
		{
			.usage = VMA_MEMORY_USAGE_AUTO
		};
		vk::Buffer destBuffer;
		VmaAllocation destBufferMem;
		vmaCreateBuffer
		(
			allocator,
			reinterpret_cast<VkBufferCreateInfo*>(&destBufferCreateInfo), &destBufferVmaAllocInfo,
			reinterpret_cast<VkBuffer*>(&destBuffer), &destBufferMem, nullptr
		);
		buffers[accessorIdx] = std::make_pair(destBuffer, destBufferMem);

		bufferSizes[accessorIdx] = vk::DeviceSize{ bufferSize };

		accessorIdx++;
	}
}

void Loader::createImages()
{
	for (size_t imageIdx = 0; tinygltf::Texture const& texture : model.textures)
	{
		tinygltf::Image const& image = model.images[texture.source];
		vk::DeviceSize bufferSize = (image.bits / 8) * image.component * image.width * image.height;
		vk::BufferCreateInfo stagingBufferInfo
		{
			{},
			bufferSize,
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::SharingMode::eExclusive,
			1,
			&queueFamilyIndex
		};
		VmaAllocationCreateInfo allocCreateInfo
		{
			.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO
		};
		vk::Buffer stagingBuffer;
		VmaAllocation stagingBufferMem;
		VmaAllocationInfo stagingBufferMemInfo;
		vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&stagingBufferInfo), &allocCreateInfo, reinterpret_cast<VkBuffer*>(&stagingBuffer), &stagingBufferMem, &stagingBufferMemInfo);
		stagingImageBuffers[imageIdx] = std::make_pair(stagingBuffer, stagingBufferMem);

		memcpy(stagingBufferMemInfo.pMappedData, image.image.data(), bufferSize);

		if (texture.sampler > 0)
		{
			tinygltf::Sampler const& sampler = model.samplers[texture.sampler];
			vk::Filter magFilter = getMagnificationFilter(sampler);
			vk::SamplerMipmapMode mipMapMode;
			vk::Filter minFilter = getMinificationFilter(sampler, mipMapMode);
			vk::SamplerAddressMode sAddressMode;
			vk::SamplerAddressMode tAddressMode;
			getSamplerAddressModes(sampler, sAddressMode, tAddressMode);
			vk::SamplerCreateInfo samplerCreateInfo
			{
				{},
				magFilter, minFilter, mipMapMode,
				sAddressMode, tAddressMode, {},
				0,
				vk::False, {},
				vk::False, {},
				0, vk::LodClampNone,
				{},
				vk::False
			};
			vk::Sampler vulkanSampler = device.createSampler(samplerCreateInfo);
			imageSamplers[imageIdx] = vulkanSampler;
		}
		else
		{ // use default sampler
			imageSamplers[imageIdx] = sampler;
		}

		vk::Format imageFormat{};
		for (tinygltf::Material mat : model.materials)
		{
			if (
				mat.emissiveTexture.index == imageIdx ||
				mat.pbrMetallicRoughness.baseColorTexture.index == imageIdx
				)
			{
				imageFormat = getSRGBImageFormat(image);
				break;
			}
			else if (
				mat.normalTexture.index == imageIdx ||
				mat.occlusionTexture.index == imageIdx ||
				mat.pbrMetallicRoughness.metallicRoughnessTexture.index == imageIdx
				)
			{
				imageFormat = getImageFormat(image);
				break;
			}
		}
		vk::ImageCreateInfo destImageCreateInfo
		{
			{},
			vk::ImageType::e2D,
			imageFormat,
			{ static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height), 1 },
			1,
			1,
			vk::SampleCountFlagBits::e1,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, // Assume all images will be used in the same way
			vk::SharingMode::eExclusive,
			1,
			&queueFamilyIndex,
			vk::ImageLayout::eUndefined
		};
		VmaAllocationCreateInfo destAllocCreateInfo
		{
			.usage = VMA_MEMORY_USAGE_AUTO
		};
		vk::Image destImage;
		VmaAllocation destImageMem;
		vmaCreateImage
		(
			allocator,
			reinterpret_cast<VkImageCreateInfo*>(&destImageCreateInfo), &destAllocCreateInfo,
			reinterpret_cast<VkImage*>(&destImage), &destImageMem, nullptr
		);
		images[imageIdx] = std::make_pair(destImage, destImageMem);

		imageExtents[imageIdx] = vk::Extent3D{ static_cast<uint32_t>(image.width), static_cast<uint32_t>(image.height), 1 };

		vk::ImageViewCreateInfo imageViewInfo{ {}, destImage, vk::ImageViewType::e2D, imageFormat, {}, { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 } };
		imageViews[imageIdx] = device.createImageView(imageViewInfo);

		imageIdx++;
	}
}

vk::Filter Loader::getMagnificationFilter(tinygltf::Sampler const& sampler)
{
	vk::Filter res;

	switch (sampler.magFilter)
	{
		case (TINYGLTF_TEXTURE_FILTER_LINEAR):
			res = vk::Filter::eLinear;
			break;
		case (TINYGLTF_TEXTURE_FILTER_NEAREST):
			[[fallthrough]];
		default:
			res = vk::Filter::eNearest;
			break;
	}

	return res;
}

vk::Filter Loader::getMinificationFilter(tinygltf::Sampler const& sampler, vk::SamplerMipmapMode& mipMapMode)
{
	vk::Filter filter;

	switch (sampler.minFilter)
	{
		case (TINYGLTF_TEXTURE_FILTER_LINEAR):
			[[fallthrough]];
		case (TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST):
			filter = vk::Filter::eLinear;
			mipMapMode = vk::SamplerMipmapMode::eNearest;
			break;
		case (TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR):
			filter = vk::Filter::eLinear;
			mipMapMode = vk::SamplerMipmapMode::eLinear;
			break;
		case (TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR):
			filter = vk::Filter::eNearest;
			mipMapMode = vk::SamplerMipmapMode::eLinear;
			break;
		case (TINYGLTF_TEXTURE_FILTER_NEAREST):
			[[fallthrough]];
		case (TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST):
			[[fallthrough]];
		default:
			filter = vk::Filter::eNearest;
			mipMapMode = vk::SamplerMipmapMode::eNearest;
			break;
	}

	return filter;
}

void Loader::getSamplerAddressModes(tinygltf::Sampler const& sampler, vk::SamplerAddressMode& sAddressMode, vk::SamplerAddressMode& tAddressMode)
{
	switch (sampler.wrapS)
	{
		case (TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE):
			sAddressMode = vk::SamplerAddressMode::eClampToEdge;
			break;
		case (TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT):
			sAddressMode = vk::SamplerAddressMode::eMirroredRepeat;
			break;
		case(TINYGLTF_TEXTURE_WRAP_REPEAT):
			[[fallthrough]];
		default:
			sAddressMode = vk::SamplerAddressMode::eRepeat;
			break;
	}
	switch (sampler.wrapT)
	{
	case (TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE):
		tAddressMode = vk::SamplerAddressMode::eClampToEdge;
		break;
	case (TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT):
		tAddressMode = vk::SamplerAddressMode::eMirroredRepeat;
		break;
	case(TINYGLTF_TEXTURE_WRAP_REPEAT):
		[[fallthrough]];
	default:
		tAddressMode = vk::SamplerAddressMode::eRepeat;
		break;
	}
}

void Loader::createNodes(std::shared_ptr<SceneNode> & sceneNode, tinygltf::Node const & node)
{
	getNodeLocalMatrix(node, sceneNode);

	tinygltf::Mesh const & mesh = model.meshes[node.mesh];
	sceneNode->surfacesDatas.reserve(mesh.primitives.size());
	for (tinygltf::Primitive const& primitive : mesh.primitives)
	{
		SurfaceData surface;

		if (primitive.mode > -1 && primitive.mode != TINYGLTF_MODE_TRIANGLES)
		{
			std::cout << "No support for non-triangle-list geometry" << std::endl;
			exit(-1);
		}
		surface.topology = vk::PrimitiveTopology::eTriangleList;

		tinygltf::Material const & material = model.materials[primitive.material];
		auto albTexInd = material.pbrMetallicRoughness.baseColorTexture.index;
		auto metRoughTexInd = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
		auto normlTexInd = material.normalTexture.index;
		//auto emisTexInd = material.emissiveTexture.index;
		//auto occlTexInd = material.occlusionTexture.index;
		surface.albedoImage = images[albTexInd].first;
		surface.albedoImageMem = images[albTexInd].second;
		surface.metalRoughImage = images[metRoughTexInd].first;
		surface.metalRoughImageMem = images[metRoughTexInd].second;
		surface.normalImage = images[normlTexInd].first;
		surface.normalImageMem = images[normlTexInd].second;
		//images[emisTexInd].first;
		//images[emisTexInd].second;
		//images[occlTexInd].first;
		//images[occlTexInd].second;
		std::vector<std::vector<std::pair<vk::Buffer*, VmaAllocation*>>> uvCoordBuffers(5);
		uvCoordBuffers[material.pbrMetallicRoughness.baseColorTexture.texCoord].push_back(std::make_pair(&surface.albedoUVBuffer, &surface.albedoUVBufferMem));
		uvCoordBuffers[material.pbrMetallicRoughness.metallicRoughnessTexture.texCoord].push_back(std::make_pair(&surface.metalRoughUVBuffer, &surface.metalRoughUVBufferMem));
		uvCoordBuffers[material.normalTexture.texCoord].push_back(std::make_pair(&surface.normalUVBuffer, &surface.normalUVBufferMem));

		for (auto & [name, index] : primitive.attributes)
		{
			if (name.compare("POSITION") == 0)
			{
				surface.positionBuffer = buffers[index].first;
				surface.positionBufferMem = buffers[index].second;
			}
			else if (name.compare("NORMAL") == 0)
			{
				surface.normalBuffer = buffers[index].first;
				surface.normalBufferMem = buffers[index].second;
			}
			else if (name.compare("TANGENT") == 0)
			{
				surface.tangentBuffer= buffers[index].first;
				surface.tangentBufferMem = buffers[index].second;
			}
			else if (name.compare("TEXCOORD_0") == 0)
			{
				for (auto [buff, buffMem] : uvCoordBuffers[0])
				{
					*buff = buffers[index].first;
					*buffMem = buffers[index].second;
				}
			}
			else if (name.compare("TEXCOORD_1") == 0)
			{
				for (auto [buff, buffMem] : uvCoordBuffers[1])
				{
					*buff = buffers[index].first;
					*buffMem = buffers[index].second;
				}
			}
			else if (name.compare("TEXCOORD_2") == 0)
			{
				for (auto [buff, buffMem] : uvCoordBuffers[2])
				{
					*buff = buffers[index].first;
					*buffMem = buffers[index].second;
				}
			}
			else if (name.compare("TEXCOORD_3") == 0)
			{
				for (auto [buff, buffMem] : uvCoordBuffers[3])
				{
					*buff = buffers[index].first;
					*buffMem = buffers[index].second;
				}
			}
			else if (name.compare("TEXCOORD_4") == 0)
			{
				for (auto [buff, buffMem] : uvCoordBuffers[4])
				{
					*buff = buffers[index].first;
					*buffMem = buffers[index].second;
				}
			}
		}

		surface.indexBuffer = buffers[primitive.indices].first;
		surface.indexBufferMem = buffers[primitive.indices].second;
		surface.indexNumber = bufferSizes[primitive.indices] / 2; // assume 16-bit(2-byte) indices

		vk::DescriptorSetAllocateInfo descriptorSetAllocationInfo{ descriptorPool, 1, &descriptorSetLayout };
		vk::DescriptorSet descriptorSet;
		auto res = device.allocateDescriptorSets(&descriptorSetAllocationInfo, &descriptorSet);
		vk::DescriptorImageInfo descriptorAlbedoImageInfo{ imageSamplers[albTexInd], imageViews[albTexInd], vk::ImageLayout::eShaderReadOnlyOptimal };
		vk::DescriptorImageInfo descriptorNormalImageInfo{ imageSamplers[normlTexInd], imageViews[normlTexInd], vk::ImageLayout::eShaderReadOnlyOptimal };
		vk::DescriptorImageInfo descriptorMetRoughImageInfo{ imageSamplers[metRoughTexInd], imageViews[metRoughTexInd], vk::ImageLayout::eShaderReadOnlyOptimal };
		std::array<vk::WriteDescriptorSet, 3> descriptorWrites;
		descriptorWrites[0] = { descriptorSet, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &descriptorAlbedoImageInfo };
		descriptorWrites[1] = { descriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &descriptorNormalImageInfo };
		descriptorWrites[2] = { descriptorSet, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &descriptorMetRoughImageInfo };
		device.updateDescriptorSets(3, descriptorWrites.data(), 0, nullptr);
		surface.descriptorSet = descriptorSet;

		sceneNode->surfacesDatas.push_back(surface);
	}

	for (int childIdx : node.children)
	{
		tinygltf::Node const & childNode = model.nodes[childIdx];
		std::shared_ptr<SceneNode> childSceneNode = std::make_shared<SceneNode>();
		childSceneNode->parent = sceneNode;
		createNodes(childSceneNode, childNode);
	}
}

void Loader::loadGLTFHelper(std::string const & file)
{
	tinygltf::TinyGLTF loader;
	std::string warn, err;
	bool res;

	loader.SetPreserveImageChannels(false);
	if (utils::hasSuffix(file, ".gltf"))
	{
		res = loader.LoadASCIIFromFile(&model, &err, &warn, file);
	}
	else if (utils::hasSuffix(file, ".glb"))
	{
		res = loader.LoadBinaryFromFile(&model, &err, &warn, file);
	}
	else
	{
		std::cout << "Model file name has neither gltf or glb extension in his name" << std::endl;
		exit(-1);
	}
	if (!res)
	{
		std::cout << "Error while loading glTF/glb file: " << err << std::endl;
		exit(-1);
	}
	if (!warn.empty())
	{
		std::cout << "Warning while loading glTF/glb file: " << warn << std::endl;
	}
}

void Loader::unpackArgs(Loader::LoaderArgs const & args)
{
	device = args.device;
	allocator = args.allocator;
	descriptorPool = args.descriptorPool;
	descriptorSetLayout = args.descriptorSetLayout;
	sampler = args.sampler;
	queue = args.queue;
	queueFamilyIndex = args.queueFamilyIndex;
	commandBuffer = args.commandBuffer;
}

vk::Format Loader::getImageFormat(tinygltf::Image const & image)
{
	vk::Format res{};

	switch (image.pixel_type)
	{
		case (TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE):
			switch (image.component)
			{
				case (4):
					[[fallthrough]];
				case (3):
					res = vk::Format::eR8G8B8A8Unorm;
					break;
				case (2):
					res = vk::Format::eR8G8Unorm;
					break;
				case (1):
					res = vk::Format::eR8Unorm;
					break;
			}
			break;
		default:
			std::cout << "Coudln't find image format for staging image when loading GLTF file" << std::endl;
			exit(-1);
			break;
	}

	return res;
}

vk::Format Loader::getSRGBImageFormat(tinygltf::Image const & image)
{
	vk::Format res{};

	switch (image.pixel_type)
	{
	case (TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE):
		switch (image.component)
		{
		case (4):
			[[fallthrough]];
		case (3):
			res = vk::Format::eR8G8B8A8Srgb;
			break;
		case (2):
			res = vk::Format::eR8G8Srgb;
			break;
		case (1):
			res = vk::Format::eR8Srgb;
			break;
		}
		break;
	default:
		std::cout << "Could not find image format for sRGB staging image when loading GLTF file" << std::endl;
		exit(-1);
		break;
	}

	return res;
}

void Loader::getNodeLocalMatrix(tinygltf::Node const & node, std::shared_ptr<SceneNode> const & sceneNode)
{
	if (!node.matrix.empty())
	{
		sceneNode->localMatrix = glm::make_mat4(node.matrix.data());
	}
	else if (!node.translation.empty() || !node.rotation.empty() || !node.scale.empty())
	{
		glm::vec3 scale = (node.scale.empty()) ? glm::highp_dvec3{ 1 } : glm::make_vec3(node.scale.data());
		glm::vec4 rotation = (node.rotation.empty()) ? glm::highp_dvec4{ 0, 0, 0, 1 } : glm::make_vec4(node.rotation.data());
		glm::mat4 R = glm::mat4_cast(glm::quat{ rotation.w, rotation.x, rotation.y, rotation.z });
		glm::vec3 translation = (node.translation.empty()) ? glm::highp_dvec3{ 0 } : glm::make_vec3(node.translation.data());

		glm::scale(sceneNode->localMatrix, scale);
		sceneNode->localMatrix = R * sceneNode->localMatrix;
		glm::translate(sceneNode->localMatrix, translation);
	}
}

size_t Loader::getBytePerComponent(size_t compTypeID)
{
	size_t res{ 0 };
	
	switch (compTypeID)
	{
		case (TINYGLTF_COMPONENT_TYPE_BYTE):
			[[fallthrough]];
		case (TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE):
			res = 1;
			break;
		case (TINYGLTF_COMPONENT_TYPE_SHORT):
			[[fallthrough]];
		case (TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT):
			res = 2;
			break;
		case (TINYGLTF_COMPONENT_TYPE_INT):
			[[fallthrough]];
		case (TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT):
			[[fallthrough]];
		case (TINYGLTF_COMPONENT_TYPE_FLOAT):
			res = 4;
			break;
		case (TINYGLTF_COMPONENT_TYPE_DOUBLE):
			res = 8;
			break;
		default:
			std::cout << "Passed parameter is not a TINYGLTF_COMPONENT_TYPE_XXX macro" << std::endl;
			exit(-1);
	}

	return res;
}

size_t Loader::getNumberOfComponents(size_t typeID)
{
	size_t res{ 0 };

	switch (typeID)
	{
		case (TINYGLTF_TYPE_SCALAR):
			res = 1;
			break;
		case (TINYGLTF_TYPE_VEC2):
			res = 2;
			break;
		case (TINYGLTF_TYPE_VEC3):
			res = 3;
			break;
		case (TINYGLTF_TYPE_VEC4):
			[[fallthrough]];
		case (TINYGLTF_TYPE_MAT2):
			res = 4;
			break;
		case (TINYGLTF_TYPE_MAT3):
			res = 9;
			break;
		case (TINYGLTF_TYPE_MAT4):
			res = 16;
			break;
		default:
			std::cout << "Passed parameter is not a TINYGLTF_TYPE_XXX macro" << std::endl;
			exit(-1);
	}

	return res;
}

size_t Loader::getAccessorElementSize(size_t compTypeID, size_t typeID)
{
	return getBytePerComponent(compTypeID) * getNumberOfComponents(typeID);
}

void Loader::setupCommandBuffer()
{
	commandBuffer.reset();
	vk::CommandBufferBeginInfo beginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
	commandBuffer.begin(beginInfo);
}

void Loader::recordCommandBuffer()
{
	for (size_t i = 0; i < buffers.size(); i++)
	{
		vk::BufferCopy copy
		{
			0, 0,
			bufferSizes[i]
		};
		commandBuffer.copyBuffer(stagingBuffers[i].first, buffers[i].first, 1, &copy);
	}

	std::vector<vk::ImageMemoryBarrier2> barriersBefore(images.size());
	for (size_t i = 0; i < images.size(); i++)
	{
		barriersBefore[i] = vk::ImageMemoryBarrier2
		{
			vk::PipelineStageFlagBits2::eTopOfPipe,
			vk::AccessFlagBits2::eNone,
			vk::PipelineStageFlagBits2::eCopy,
			vk::AccessFlagBits2::eTransferWrite,
			vk::ImageLayout::eUndefined,
			vk::ImageLayout::eTransferDstOptimal,
			queueFamilyIndex,
			queueFamilyIndex,
			images[i].first,
			vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
		};
	}
	vk::DependencyInfo depInfoBefore
	{
		{},
		0, nullptr,
		0, nullptr,
		static_cast<uint32_t>(barriersBefore.size()), barriersBefore.data()
	};
	commandBuffer.pipelineBarrier2(depInfoBefore);

	for (size_t i = 0; i < images.size(); i++)
	{
		vk::BufferImageCopy copy
		{
			0,
			0, 0,
			vk::ImageSubresourceLayers{ vk::ImageAspectFlagBits::eColor, 0, 0, 1 },
			vk::Offset3D{0, 0, 0},
			imageExtents[i]
		};
		commandBuffer.copyBufferToImage(stagingImageBuffers[i].first, images[i].first, vk::ImageLayout::eTransferDstOptimal, 1, &copy);
	}

	std::vector<vk::ImageMemoryBarrier2> barriersAfter(images.size());
	for (size_t i = 0; i < images.size(); i++)
	{
		barriersAfter[i] = vk::ImageMemoryBarrier2
		{
			vk::PipelineStageFlagBits2::eCopy,
			vk::AccessFlagBits2::eTransferWrite,
			vk::PipelineStageFlagBits2::eBottomOfPipe,
			vk::AccessFlagBits2::eNone,
			vk::ImageLayout::eTransferDstOptimal,
			vk::ImageLayout::eShaderReadOnlyOptimal,
			queueFamilyIndex,
			queueFamilyIndex,
			images[i].first,
			vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
		};
	}
	vk::DependencyInfo depInfoAfter
	{
		{},
		0, nullptr,
		0, nullptr,
		static_cast<uint32_t>(barriersAfter.size()), barriersAfter.data()
	};
	commandBuffer.pipelineBarrier2(depInfoAfter);
}

void Loader::submitCommandBuffer()
{
	commandBuffer.end();
	vk::SubmitInfo submitInfo{ 0, nullptr, nullptr, 1, &commandBuffer, 0, nullptr};
	auto res = queue.submit(1, &submitInfo, nullptr);
}

void Loader::freeStagingMemory()
{
	for (auto& imagePair : stagingImageBuffers)
	{
		vmaDestroyBuffer(allocator, imagePair.first, imagePair.second);
	}
	for (auto& bufferPair : stagingBuffers)
	{
		vmaDestroyBuffer(allocator, bufferPair.first, bufferPair.second);
	}
}