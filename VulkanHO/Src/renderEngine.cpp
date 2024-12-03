#define VMA_IMPLEMENTATION
#include "renderEngine.h"

#include "utils.hpp"
#include <fstream>
#include <chrono>
#include "loader.h"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

void RenderEngine::init()
{
	vkb::Instance instance;
	vkb::PhysicalDevice physicalDevice;
	vkb::Device device;

	createWindow();
	createVkInstance(instance);
	glfwCreateWindowSurface(vkInstance, window, nullptr, reinterpret_cast<VkSurfaceKHR*>(&vkSurface));
	pickVkPhysicalDevice(instance, physicalDevice);
	createVkDevice(physicalDevice, device);
	initVkHppDispatcher();
	getQueues(device);
	createSwapchain(device);
	createAllocator();
	createSamplers();
	createCommandPools();
	createDescriptorPools();
	createDescriptorSetLayouts();
	createFramesResources();
	createCommandBuffers();
	createSyncStructures();
	loadModels();
	createPipelines();
}

void RenderEngine::createWindow()
{
	const char* errMsg;
	if (!glfwInit())
	{
		glfwGetError(&errMsg);
		std::cout << errMsg << std::endl;
		exit(-1);
	}
	glfwWindowHint(GLFW_CLIENT_API ,GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
	window = glfwCreateWindow(windowWidth, windowHeight, "Renderer", nullptr, nullptr);
	if (!window)
	{
		glfwGetError(&errMsg);
		std::cout << errMsg << std::endl;
		exit(-1);
	}
}

void RenderEngine::createVkInstance(vkb::Instance & instance)
{
	vkb::InstanceBuilder instanceBuilder{};

	instanceBuilder.set_app_name("Rendering App")
		.set_app_version(VK_MAKE_API_VERSION(0, 1, 0, 0))
		.set_engine_name("Renderer")
		.set_engine_version(VK_MAKE_API_VERSION(0, 1, 0, 0))
		.require_api_version(1, 3, 0);
#ifdef _DEBUG
	instanceBuilder.enable_validation_layers()
		.use_default_debug_messenger();
#endif

	uint32_t reqExtsNum;
	auto reqExts = glfwGetRequiredInstanceExtensions(&reqExtsNum);

	instanceBuilder.enable_extensions(reqExtsNum, reqExts);

	auto instanceRes = instanceBuilder.build();
	utils::checkVkbError(instanceRes);
	instance = instanceRes.value();
	vkInstance = instance;
	vkDebugMess = instance.debug_messenger;
}

void RenderEngine::pickVkPhysicalDevice(const vkb::Instance & instance, vkb::PhysicalDevice & physicalDevice)
{
	vkb::PhysicalDeviceSelector physDevSelector(instance, vkSurface);

	vk::PhysicalDeviceFeatures features{};
	features.independentBlend = vk::True;

	vk::PhysicalDeviceVulkan13Features features13{};
	features13.dynamicRendering = vk::True;
	features13.synchronization2 = vk::True;

	physDevSelector.set_minimum_version(1, 3)
		.set_required_features(features)
		.set_required_features_13(features13)
		.add_required_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)
		.add_required_extension(VK_NV_DESCRIPTOR_POOL_OVERALLOCATION_EXTENSION_NAME);

	auto res = physDevSelector.select();
	utils::checkVkbError(res);
	physicalDevice = res.value();
	vkPhysicalDevice = physicalDevice;
}

void RenderEngine::createVkDevice(const vkb::PhysicalDevice & physicalDevice, vkb::Device & device)
{
	vkb::DeviceBuilder devBuilder(physicalDevice);

	auto res = devBuilder.build();
	utils::checkVkbError(res);
	device = res.value();
	vkDevice = device;
}

void RenderEngine::initVkHppDispatcher()
{
	VULKAN_HPP_DEFAULT_DISPATCHER.init();
	VULKAN_HPP_DEFAULT_DISPATCHER.init(vkInstance);
	VULKAN_HPP_DEFAULT_DISPATCHER.init(vkDevice);
}

void RenderEngine::getQueues(const vkb::Device & device)
{
	auto qres = device.get_queue(vkb::QueueType::graphics);
	utils::checkVkbError(qres);
	graphicsQueue = qres.value();

	auto ires = device.get_queue_index(vkb::QueueType::graphics);
	utils::checkVkbError(ires);
	graphicsQueueIdx = ires.value();
}

void RenderEngine::createSwapchain(const vkb::Device & device)
{
	vkb::SwapchainBuilder swapchainBuilder(device, vkSurface);

	swapchainBuilder.set_required_min_image_count(3)
		.set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR);

	auto res = swapchainBuilder.build();
	utils::checkVkbError(res);
	vkb::Swapchain vkbSwapchain = res.value();
	vkSwapchain = vkbSwapchain;

	swapchainImageExtent = vkbSwapchain.extent;
	swapchainImageFormat = static_cast<vk::Format>(vkbSwapchain.image_format);

	auto resImg = vkbSwapchain.get_images();
	utils::checkVkbError(resImg);
	auto resImgV = vkbSwapchain.get_image_views();
	utils::checkVkbError(resImgV);

	swapchainImgs.reserve(vkbSwapchain.image_count);
	swapchainImgsViews.reserve(vkbSwapchain.image_count);
	for (auto const & img : resImg.value())
	{
		swapchainImgs.push_back(img);
	}
	for (auto const& imgV : resImgV.value())
	{
		swapchainImgsViews.push_back(imgV);
	}
}

void RenderEngine::createAllocator()
{
	VmaAllocatorCreateInfo allocatorInfo
	{
		.physicalDevice = vkPhysicalDevice,
		.device = vkDevice,
		.instance = vkInstance,
		.vulkanApiVersion = VK_API_VERSION_1_3
	};

	vmaCreateAllocator(&allocatorInfo, &allocator);
}

void RenderEngine::createSamplers()
{
	vk::SamplerCreateInfo samplInfo{ {}, vk::Filter::eNearest, vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat };
	nearestSampler = vkDevice.createSampler(samplInfo);

	samplInfo.magFilter = vk::Filter::eLinear;
	samplInfo.minFilter = vk::Filter::eLinear;
	samplInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
	linearSampler = vkDevice.createSampler(samplInfo);
}

void RenderEngine::createCommandPools()
{
	vk::CommandPoolCreateInfo commPoolInfo{ vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphicsQueueIdx, nullptr};

	mainCommPool = vkDevice.createCommandPool(commPoolInfo);
}

void RenderEngine::createFramesResources()
{
	vk::ImageCreateInfo imageInfo{ {}, vk::ImageType::e2D, {}, vk::Extent3D{swapchainImageExtent, 1}, 1, 1,  vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled, vk::SharingMode::eExclusive, 1, &graphicsQueueIdx, {} };
	vk::ImageCreateInfo depthImageInfo{ {}, vk::ImageType::e2D, vk::Format::eD32Sfloat, vk::Extent3D{swapchainImageExtent, 1}, 1, 1,  vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled, vk::SharingMode::eExclusive, 1, &graphicsQueueIdx, {} };
	vk::ImageSubresourceRange subresRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
	vk::ImageSubresourceRange depthSubresRange{ vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 };
	vk::ImageViewCreateInfo imageViewInfo{ {}, {}, vk::ImageViewType::e2D, {}, {}, subresRange };
	vk::ImageViewCreateInfo depthImageViewInfo{ {}, {}, vk::ImageViewType::e2D, vk::Format::eD32Sfloat, {}, depthSubresRange };
	VmaAllocationCreateInfo allocInfo{ .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, .usage = VMA_MEMORY_USAGE_AUTO };

	vk::CommandBufferAllocateInfo commBuffAllocInfo{ mainCommPool,  vk::CommandBufferLevel::ePrimary, 1 };

	vk::SemaphoreCreateInfo semaphInfo{};
	vk::FenceCreateInfo fenceInfo{ vk::FenceCreateFlagBits::eSignaled };

	for (FrameResources& fr : framesResources)
	{
		imageInfo.format = vk::Format::eR32G32B32A32Sfloat;
		imageViewInfo.format = vk::Format::eR32G32B32A32Sfloat;

		vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&imageInfo), &allocInfo, reinterpret_cast<VkImage*>(&fr.posImage), &fr.posImageMem, nullptr);
		imageViewInfo.image = fr.posImage;
		fr.posImageView = vkDevice.createImageView(imageViewInfo);

		imageInfo.format = vk::Format::eR16G16B16A16Unorm;
		imageViewInfo.format = vk::Format::eR16G16B16A16Unorm;

		vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&imageInfo), &allocInfo, reinterpret_cast<VkImage*>(&fr.albedoImage), &fr.albedoImageMem, nullptr);
		imageViewInfo.image = fr.albedoImage;
		fr.albedoImageView = vkDevice.createImageView(imageViewInfo);

		imageInfo.format = vk::Format::eR16G16B16A16Snorm;
		imageViewInfo.format = vk::Format::eR16G16B16A16Snorm;

		vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&imageInfo), &allocInfo, reinterpret_cast<VkImage*>(&fr.normalImage), &fr.normalImageMem, nullptr);
		imageViewInfo.image = fr.normalImage;
		fr.normalImageView = vkDevice.createImageView(imageViewInfo);

		imageInfo.format = vk::Format::eR16G16Unorm;
		imageViewInfo.format = vk::Format::eR16G16Unorm;

		vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&imageInfo), &allocInfo, reinterpret_cast<VkImage*>(&fr.metalRoughImage), &fr.metalRoughImageMem, nullptr);
		imageViewInfo.image = fr.metalRoughImage;
		fr.metalRoughImageView = vkDevice.createImageView(imageViewInfo);

		vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&depthImageInfo), &allocInfo, reinterpret_cast<VkImage*>(&fr.depthImage), &fr.depthImageMem, nullptr);
		depthImageViewInfo.image = fr.depthImage;
		fr.depthImageView = vkDevice.createImageView(depthImageViewInfo);


		auto res = vkDevice.allocateCommandBuffers(&commBuffAllocInfo, &fr.commBuff);
		if (res != vk::Result::eSuccess)
		{
			std::cout << "Failed to create command buffer for a frame" << std::endl;
			exit(-1);
		}


		fr.swapchainSem = vkDevice.createSemaphore(semaphInfo);
		fr.renderSem = vkDevice.createSemaphore(semaphInfo);
		fr.renderFence = vkDevice.createFence(fenceInfo);
	}
}

void RenderEngine::createDescriptorPools()
{
	vk::DescriptorPoolSize modelPoolSize{ vk::DescriptorType::eCombinedImageSampler, 3 };
	vk::DescriptorPoolCreateInfo modelPoolInfo{ vk::DescriptorPoolCreateFlagBits::eAllowOverallocationSetsNV | vk::DescriptorPoolCreateFlagBits::eAllowOverallocationPoolsNV | vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 10, 1, &modelPoolSize };

	modelsDescriptorPool = vkDevice.createDescriptorPool(modelPoolInfo);
}

void RenderEngine::createDescriptorSetLayouts()
{
	createGeometricDescriptorLayout();
	createRenderDescriptorLayout();
}

void RenderEngine::createGeometricDescriptorLayout()
{
	std::vector<vk::DescriptorSetLayoutBinding> bindings(3);
	bindings[0] = vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
	bindings[1] = vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
	bindings[2] = vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};

	vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{ {}, 3, bindings.data()};

	geometricDescriptorSetLayout = vkDevice.createDescriptorSetLayout(descriptorSetLayoutInfo, nullptr);
}

void RenderEngine::createRenderDescriptorLayout()
{
	std::vector<vk::DescriptorSetLayoutBinding> bindings(5);
	bindings[0] = vk::DescriptorSetLayoutBinding{ 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
	bindings[1] = vk::DescriptorSetLayoutBinding{ 1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
	bindings[2] = vk::DescriptorSetLayoutBinding{ 2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
	bindings[3] = vk::DescriptorSetLayoutBinding{ 3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};
	bindings[4] = vk::DescriptorSetLayoutBinding{ 4, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment};

	vk::DescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{ vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR, 5, bindings.data()};
	
	renderDescriptorSetLayout = vkDevice.createDescriptorSetLayout(descriptorSetLayoutInfo, nullptr);
}

void RenderEngine::createCommandBuffers()
{

}

void RenderEngine::createSyncStructures()
{

}

void RenderEngine::loadModels()
{
	Loader loader;
	Loader::LoaderArgs loaderArgs
	{
		.device = vkDevice,
		.allocator = allocator,
		.descriptorPool = modelsDescriptorPool,
		.descriptorSetLayout = geometricDescriptorSetLayout,
		.sampler = linearSampler,
		.queue = graphicsQueue,
		.commandBuffer = framesResources[0].commBuff,
		.queueFamilyIndex = graphicsQueueIdx
	};
	sceneNodes.push_back(loader.loadGLTF(loaderArgs, "Models/WaterBottle.glb"));
}

void RenderEngine::createPipelines()
{
	createGeometricPipeline();
	createRenderPipeline();
}

void RenderEngine::createGeometricPipeline()
{
	std::ifstream shaderFile;
	shaderFile.open("Shaders/geom.vert.spv", std::ios::ate | std::ios::binary);
	if (!shaderFile.is_open())
	{
		std::cout << "Geometric vertex shader not found" << std::endl;
		exit(-1);
	}

	size_t shaderFileSize = shaderFile.tellg();

	shaderFile.seekg(0);

	std::vector<uint32_t> shaderFileBinary(shaderFileSize / sizeof(uint32_t));

	shaderFile.read(reinterpret_cast<char*>(shaderFileBinary.data()), shaderFileSize);

	shaderFile.close();

	vk::ShaderModuleCreateInfo shadModCreInfo{ {}, shaderFileSize, shaderFileBinary.data()};

	vk::ShaderModule vertexShaderModule = vkDevice.createShaderModule(shadModCreInfo, nullptr);

	shaderFile.open("Shaders/geom.frag.spv", std::ios::ate | std::ios::binary);
	if (!shaderFile.is_open())
	{
		std::cout << "Geometric fragment shader not found" << std::endl;
		exit(-1);
	}

	shaderFileSize = shaderFile.tellg();

	shaderFile.seekg(0);

	shaderFileBinary.resize(shaderFileSize / sizeof(uint32_t));

	shaderFile.read(reinterpret_cast<char*>(shaderFileBinary.data()), shaderFileSize);

	shaderFile.close();

	shadModCreInfo.codeSize = shaderFileSize;
	shadModCreInfo.pCode = shaderFileBinary.data();

	vk::ShaderModule fragmentShaderModule = vkDevice.createShaderModule(shadModCreInfo, nullptr);
	
	std::vector<vk::PipelineShaderStageCreateInfo> stagesInfo(2);
	stagesInfo[0] = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eVertex, vertexShaderModule, "main"};
	stagesInfo[1] = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eFragment, fragmentShaderModule, "main"};

	std::array<vk::VertexInputBindingDescription, 6> bindDescrips;
	bindDescrips[0] = vk::VertexInputBindingDescription{ 0, sizeof(glm::vec3), vk::VertexInputRate::eVertex };
	bindDescrips[1] = vk::VertexInputBindingDescription{ 1, sizeof(glm::vec3), vk::VertexInputRate::eVertex };
	bindDescrips[2] = vk::VertexInputBindingDescription{ 2, sizeof(glm::vec4), vk::VertexInputRate::eVertex };
	bindDescrips[3] = vk::VertexInputBindingDescription{ 3, sizeof(glm::vec2), vk::VertexInputRate::eVertex };
	bindDescrips[4] = vk::VertexInputBindingDescription{ 4, sizeof(glm::vec2), vk::VertexInputRate::eVertex };
	bindDescrips[5] = vk::VertexInputBindingDescription{ 5, sizeof(glm::vec2), vk::VertexInputRate::eVertex };

	std::array<vk::VertexInputAttributeDescription, 6> attribDescrips;
	attribDescrips[0] = vk::VertexInputAttributeDescription{ 0, 0, vk::Format::eR32G32B32Sfloat, 0 };
	attribDescrips[1] = vk::VertexInputAttributeDescription{ 1, 1, vk::Format::eR32G32B32Sfloat, 0 };
	attribDescrips[2] = vk::VertexInputAttributeDescription{ 2, 2, vk::Format::eR32G32B32A32Sfloat, 0 };
	attribDescrips[3] = vk::VertexInputAttributeDescription{ 3, 3, vk::Format::eR32G32Sfloat, 0 };
	attribDescrips[4] = vk::VertexInputAttributeDescription{ 4, 4, vk::Format::eR32G32Sfloat, 0 };
	attribDescrips[5] = vk::VertexInputAttributeDescription{ 5, 5, vk::Format::eR32G32Sfloat, 0 };

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{ {}, static_cast<uint32_t>(bindDescrips.size()), bindDescrips.data(), static_cast<uint32_t>(attribDescrips.size()), attribDescrips.data(), nullptr};

	vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo{ {}, vk::PrimitiveTopology::eTriangleList, vk::False, nullptr };

	vk::PipelineViewportStateCreateInfo viewInfo{ {}, 1, nullptr, 1, nullptr};

	// No culling for the moment
	vk::PipelineRasterizationStateCreateInfo rastInfo{ {}, vk::False, vk::False, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise, vk::False, {}, {}, {}, 1 };

	vk::PipelineMultisampleStateCreateInfo multiInfo{ {}, vk::SampleCountFlagBits::e1, vk::False };

	vk::PipelineDepthStencilStateCreateInfo depthStencilInfo{ {}, vk::True, vk::True, vk::CompareOp::eLessOrEqual, vk::False, vk::False, {}, {}, 0.0f, 1.0f };

	std::vector<vk::PipelineColorBlendAttachmentState> blendAttachments(4);
	blendAttachments[0] = vk::PipelineColorBlendAttachmentState{ vk::False, {}, {}, {}, {}, {}, {}, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
	blendAttachments[1] = vk::PipelineColorBlendAttachmentState{ vk::False, {}, {}, {}, {}, {}, {}, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
	blendAttachments[2] = vk::PipelineColorBlendAttachmentState{ vk::False, {}, {}, {}, {}, {}, {}, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };
	blendAttachments[3] = vk::PipelineColorBlendAttachmentState{ vk::False, {}, {}, {}, {}, {}, {}, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG };

	vk::PipelineColorBlendStateCreateInfo blendInfo{ {}, vk::False, {}, 4,  blendAttachments.data()};

	std::vector<vk::DynamicState> dynStates(2);
	dynStates[0] = vk::DynamicState::eViewport;
	dynStates[1] = vk::DynamicState::eScissor;

	vk::PipelineDynamicStateCreateInfo dynamicInfo{ {}, 2, dynStates.data() };

	std::vector<vk::Format> attachFormats(4);
	attachFormats[0] = vk::Format::eR32G32B32A32Sfloat;
	attachFormats[1] = vk::Format::eR16G16B16A16Unorm;
	attachFormats[2] = vk::Format::eR16G16B16A16Snorm;
	attachFormats[3] = vk::Format::eR16G16Unorm;

	vk::PipelineRenderingCreateInfo renderInfo{ {}, 4, attachFormats.data(), vk::Format::eD32Sfloat};

	vk::PushConstantRange pushConstantRange{ vk::ShaderStageFlagBits::eVertex, 0, sizeof(MatrixData)};

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ {}, 1, &geometricDescriptorSetLayout, 1, &pushConstantRange };

	geometricPipelineLayout = vkDevice.createPipelineLayout(pipelineLayoutInfo, nullptr);

	vk::GraphicsPipelineCreateInfo geomPipInfo{ {}, 2, stagesInfo.data(), &vertexInputInfo, &inputAssemblyInfo, nullptr, &viewInfo, &rastInfo, &multiInfo, &depthStencilInfo, &blendInfo, &dynamicInfo, geometricPipelineLayout, {}, 0, {}, 0, &renderInfo };

	auto res = vkDevice.createGraphicsPipeline({}, geomPipInfo, nullptr);
	if (res.result != vk::Result::eSuccess)
	{
		std::cout << "Failed to create geometric pipeline" << std::endl;
		exit(-1);
	}
	geometricPipeline = res.value;

	vkDevice.destroyShaderModule(vertexShaderModule);
	vkDevice.destroyShaderModule(fragmentShaderModule);
}

void RenderEngine::createRenderPipeline()
{
	std::ifstream shaderFile;
	shaderFile.open("Shaders/rend.vert.spv", std::ios::ate | std::ios::binary);
	if (!shaderFile.is_open())
	{
		std::cout << "Render vertex shader not found" << std::endl;
		exit(-1);
	}

	size_t shaderFileSize = shaderFile.tellg();

	shaderFile.seekg(0);

	std::vector<uint32_t> shaderFileBinary(shaderFileSize / sizeof(uint32_t));

	shaderFile.read(reinterpret_cast<char*>(shaderFileBinary.data()), shaderFileSize);

	shaderFile.close();

	vk::ShaderModuleCreateInfo shadModCreInfo{ {}, shaderFileSize, shaderFileBinary.data() };

	vk::ShaderModule vertexShaderModule = vkDevice.createShaderModule(shadModCreInfo, nullptr);

	shaderFile.open("Shaders/rend.frag.spv", std::ios::ate | std::ios::binary);
	if (!shaderFile.is_open())
	{
		std::cout << "Render fragment shader not found" << std::endl;
		exit(-1);
	}

	shaderFileSize = shaderFile.tellg();

	shaderFile.seekg(0);

	shaderFileBinary.resize(shaderFileSize / sizeof(uint32_t));

	shaderFile.read(reinterpret_cast<char*>(shaderFileBinary.data()), shaderFileSize);

	shaderFile.close();

	shadModCreInfo.codeSize = shaderFileSize;
	shadModCreInfo.pCode = shaderFileBinary.data();

	vk::ShaderModule fragmentShaderModule = vkDevice.createShaderModule(shadModCreInfo, nullptr);

	std::vector<vk::PipelineShaderStageCreateInfo> stagesInfo(2);
	stagesInfo[0] = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eVertex, vertexShaderModule, "main" };
	stagesInfo[1] = vk::PipelineShaderStageCreateInfo{ {}, vk::ShaderStageFlagBits::eFragment, fragmentShaderModule, "main" };

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{ {}, 0, nullptr, 0, nullptr, nullptr };

	vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo{ {}, vk::PrimitiveTopology::eTriangleList, vk::False, nullptr };

	vk::PipelineViewportStateCreateInfo viewInfo{ {}, 1, nullptr, 1, nullptr };

	vk::PipelineRasterizationStateCreateInfo rastInfo{ {}, vk::False, vk::False, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise, vk::False, {}, {}, {}, 1 };

	vk::PipelineMultisampleStateCreateInfo multiInfo{ {}, vk::SampleCountFlagBits::e1, vk::False };

	vk::PipelineDepthStencilStateCreateInfo depthStencilInfo{ {}, vk::False, vk::False, vk::CompareOp::eNever, vk::False, vk::False };

	vk::PipelineColorBlendAttachmentState blendAttachment{ vk::False, {}, {}, {}, {}, {}, {}, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA };

	vk::PipelineColorBlendStateCreateInfo blendInfo{ {}, vk::False, {}, 1,  &blendAttachment };

	std::vector<vk::DynamicState> dynStates(2);
	dynStates[0] = vk::DynamicState::eViewport;
	dynStates[1] = vk::DynamicState::eScissor;

	vk::PipelineDynamicStateCreateInfo dynamicInfo{ {}, 2, dynStates.data() };

	vk::PipelineRenderingCreateInfo renderInfo{ {}, 1, &swapchainImageFormat};

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ {}, 1, &renderDescriptorSetLayout};

	renderPipelineLayout = vkDevice.createPipelineLayout(pipelineLayoutInfo, nullptr);

	vk::GraphicsPipelineCreateInfo renderPipInfo{ {}, 2, stagesInfo.data(), &vertexInputInfo, &inputAssemblyInfo, nullptr, &viewInfo, &rastInfo, &multiInfo, &depthStencilInfo, &blendInfo, &dynamicInfo, renderPipelineLayout, {}, 0, {}, 0, &renderInfo };

	auto res = vkDevice.createGraphicsPipeline({}, renderPipInfo, nullptr);
	if (res.result != vk::Result::eSuccess)
	{
		std::cout << "Failed to create geometric pipeline" << std::endl;
		exit(-1);
	}
	renderPipeline = res.value;

	vkDevice.destroyShaderModule(vertexShaderModule);
	vkDevice.destroyShaderModule(fragmentShaderModule);
}

void RenderEngine::draw()
{
	FrameResources& fr = getNextFrameResources();

	auto res = vkDevice.waitForFences(1, &fr.renderFence, vk::True, UINT64_MAX);
	res = vkDevice.resetFences(1, &fr.renderFence);

	fr.commBuff.reset();

	auto resSwap = vkDevice.acquireNextImageKHR(vkSwapchain, UINT64_MAX, fr.swapchainSem);
	if (resSwap.result != vk::Result::eSuccess)
	{
		std::cout << "Failed to acquire image from swapchain" << std::endl;
		exit(-1);
	}
	uint32_t swapchainImgIdx = resSwap.value;

	vk::CommandBufferBeginInfo beginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit };
	fr.commBuff.begin(beginInfo);

	changeImagesLayoutForGeometric(fr);
	drawGeometric(fr);
	changeImagesLayoutForRender(fr, swapchainImgIdx);
	drawRender(fr, swapchainImgIdx);
	changeImageLayoutForPresent(fr, swapchainImgIdx);

	fr.commBuff.end();

	vk::CommandBufferSubmitInfo commBuffSubmitInfo{ fr.commBuff };
	vk::SemaphoreSubmitInfo swapchainSemSubmitInfo{ fr.swapchainSem, {}, vk::PipelineStageFlagBits2::eColorAttachmentOutput };
	vk::SemaphoreSubmitInfo renderSemSubmitInfo{ fr.renderSem, {}, vk::PipelineStageFlagBits2::eBottomOfPipe };
	vk::SubmitInfo2 submitInfo{ {}, 1, &swapchainSemSubmitInfo, 1, &commBuffSubmitInfo, 1, &renderSemSubmitInfo};
	res = graphicsQueue.submit2(1, &submitInfo, fr.renderFence);

	vk::PresentInfoKHR presentInfo{ 1, &fr.renderSem, 1, &vkSwapchain, &swapchainImgIdx};
	res = graphicsQueue.presentKHR(presentInfo);
}

void RenderEngine::changeImagesLayoutForGeometric(FrameResources& frameResources)
{
	vk::ImageSubresourceRange imgRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	vk::ImageSubresourceRange depthImgRange{ vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
	std::vector<vk::ImageMemoryBarrier2> barriers(5);
	barriers[0] = { vk::PipelineStageFlagBits2::eTopOfPipe, {}, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, graphicsQueueIdx, graphicsQueueIdx, frameResources.posImage, imgRange };
	barriers[1] = { vk::PipelineStageFlagBits2::eTopOfPipe, {}, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, graphicsQueueIdx, graphicsQueueIdx, frameResources.albedoImage, imgRange };
	barriers[2] = { vk::PipelineStageFlagBits2::eTopOfPipe, {}, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, graphicsQueueIdx, graphicsQueueIdx, frameResources.normalImage, imgRange };
	barriers[3] = { vk::PipelineStageFlagBits2::eTopOfPipe, {}, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, graphicsQueueIdx, graphicsQueueIdx, frameResources.metalRoughImage, imgRange };
	barriers[4] = { vk::PipelineStageFlagBits2::eTopOfPipe, {}, vk::PipelineStageFlagBits2::eEarlyFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthAttachmentOptimal, graphicsQueueIdx, graphicsQueueIdx, frameResources.depthImage, depthImgRange };
	vk::DependencyInfo depInfo{ {}, 0, nullptr, 0, nullptr, 5, barriers.data()};

	frameResources.commBuff.pipelineBarrier2(depInfo);
}

void RenderEngine::drawGeometric(FrameResources& frameResources)
{
	vk::Rect2D rendArea{ {}, swapchainImageExtent };
	vk::ClearColorValue albedoClearVal{ 1.0f, 0.0f, 1.0f, 1.0f };
	vk::ClearColorValue colorClearVal{ 0.0f, 0.0f, 0.0f, 1.0f };
	vk::ClearDepthStencilValue depthClearVal{ 1.0f };
	std::vector<vk::RenderingAttachmentInfo> colorAttachments(4);
	colorAttachments[0] = { frameResources.posImageView, vk::ImageLayout::eColorAttachmentOptimal, {}, {}, {}, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, { colorClearVal } };
	colorAttachments[1] = { frameResources.albedoImageView, vk::ImageLayout::eColorAttachmentOptimal, {}, {}, {}, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, { albedoClearVal } };
	colorAttachments[2] = { frameResources.normalImageView, vk::ImageLayout::eColorAttachmentOptimal, {}, {}, {}, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, { colorClearVal } };
	colorAttachments[3] = { frameResources.metalRoughImageView, vk::ImageLayout::eColorAttachmentOptimal, {}, {}, {}, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, { colorClearVal } };
	vk::RenderingAttachmentInfo depthAttachment{ frameResources.depthImageView, vk::ImageLayout::eDepthAttachmentOptimal, {}, {}, {}, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, { depthClearVal } };
	vk::RenderingInfo rendInfo{ {}, rendArea, 1, 0, 4, colorAttachments.data(), &depthAttachment};

	frameResources.commBuff.beginRendering(rendInfo);

	frameResources.commBuff.bindPipeline(vk::PipelineBindPoint::eGraphics, geometricPipeline);

	vk::Viewport viewport{ 0, 0, static_cast<float>(swapchainImageExtent.width), static_cast<float>(swapchainImageExtent.height), 0, 1 };
	vk::Rect2D scissor{ {}, swapchainImageExtent };
	frameResources.commBuff.setViewport(0, viewport);
	frameResources.commBuff.setScissor(0, scissor);

	MatrixData matrixData;
	glm::mat4 viewMat = glm::lookAt(glm::vec3{ 0.0f, 0.0f, 0.5f }, glm::vec3{ 0.0f, 0.0f, 0.0f }, glm::vec3{ 0.0f, 1.0f, 0.0f });
	glm::mat4 modelMat = glm::rotate(glm::scale(glm::mat4{ 1.0f }, { 1.0f, 1.0f, 1.0f }), glm::radians(-30.0f), { 1.0f, 0.0f, 0.0f });
	matrixData.MV = viewMat * modelMat;
	matrixData.P = utils::glmVkPerspective(glm::radians(60.0f), static_cast<float>(windowWidth) / windowHeight, 0.1f, 50.0f);
	frameResources.commBuff.pushConstants(geometricPipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, sizeof(MatrixData), &matrixData);

	for (auto& sceneNode : sceneNodes)
	{
		for (auto& surface : sceneNode->surfacesDatas)
		{
			frameResources.commBuff.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, geometricPipelineLayout, 0, 1, &surface.descriptorSet, 0, nullptr);
			frameResources.commBuff.bindIndexBuffer(surface.indexBuffer, 0, vk::IndexType::eUint16);
			std::array<vk::Buffer, 6> buffers;
			buffers[0] = surface.positionBuffer;
			buffers[1] = surface.normalBuffer;
			buffers[2] = surface.tangentBuffer;
			buffers[3] = surface.albedoUVBuffer;
			buffers[4] = surface.normalUVBuffer;
			buffers[5] = surface.metalRoughUVBuffer;
			std::array<vk::DeviceSize, 6> offsets;
			offsets[0] = vk::DeviceSize{ 0 };
			offsets[1] = vk::DeviceSize{ 0 };
			offsets[2] = vk::DeviceSize{ 0 };
			offsets[3] = vk::DeviceSize{ 0 };
			offsets[4] = vk::DeviceSize{ 0 };
			offsets[5] = vk::DeviceSize{ 0 };
			frameResources.commBuff.bindVertexBuffers(0, static_cast<uint32_t>(buffers.size()), buffers.data(), offsets.data());
			frameResources.commBuff.drawIndexed(surface.indexNumber, 1, 0, 0, 0);
		}
	}

	frameResources.commBuff.endRendering();
}

void RenderEngine::changeImagesLayoutForRender(FrameResources& frameResources, uint32_t swapchainImageIndex)
{
	vk::ImageSubresourceRange imgRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
	vk::ImageSubresourceRange depthImgRange{ vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1 };
	std::vector<vk::ImageMemoryBarrier2> barriers(6);
	barriers[0] = { vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, graphicsQueueIdx, graphicsQueueIdx, frameResources.posImage, imgRange };
	barriers[1] = { vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, graphicsQueueIdx, graphicsQueueIdx, frameResources.albedoImage, imgRange };
	barriers[2] = { vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, graphicsQueueIdx, graphicsQueueIdx, frameResources.normalImage, imgRange };
	barriers[3] = { vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, graphicsQueueIdx, graphicsQueueIdx, frameResources.metalRoughImage, imgRange };
	barriers[4] = { vk::PipelineStageFlagBits2::eLateFragmentTests, vk::AccessFlagBits2::eDepthStencilAttachmentWrite, vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderSampledRead, vk::ImageLayout::eDepthAttachmentOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, graphicsQueueIdx, graphicsQueueIdx, frameResources.depthImage, depthImgRange };
	barriers[5] = { vk::PipelineStageFlagBits2::eTopOfPipe, {}, vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, graphicsQueueIdx, graphicsQueueIdx, swapchainImgs[swapchainImageIndex], imgRange };
	vk::DependencyInfo depInfo{ {}, 0, nullptr, 0, nullptr, 6, barriers.data() };

	frameResources.commBuff.pipelineBarrier2(depInfo);
}

void RenderEngine::drawRender(FrameResources& frameResources, uint32_t swapchainImageIndex)
{
	vk::Rect2D rendArea{ {}, swapchainImageExtent };
	vk::RenderingAttachmentInfo attachInfo{ swapchainImgsViews[swapchainImageIndex], vk::ImageLayout::eColorAttachmentOptimal, {}, {}, {}, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eStore, {} };
	vk::RenderingInfo rendInfo{ {}, rendArea, 1, 0, 1, &attachInfo};

	frameResources.commBuff.beginRendering(rendInfo);

	frameResources.commBuff.bindPipeline( vk::PipelineBindPoint::eGraphics, renderPipeline);

	vk::Viewport viewport{ 0, 0, static_cast<float>(swapchainImageExtent.width), static_cast<float>(swapchainImageExtent.height), 0, 1 };
	vk::Rect2D scissor{ {}, swapchainImageExtent };
	frameResources.commBuff.setViewport(0, viewport);
	frameResources.commBuff.setScissor(0, scissor);

	std::vector<vk::DescriptorImageInfo> dsImgInfos(5);
	dsImgInfos[0] = { linearSampler, frameResources.posImageView, vk::ImageLayout::eShaderReadOnlyOptimal };
	dsImgInfos[1] = { linearSampler, frameResources.albedoImageView, vk::ImageLayout::eShaderReadOnlyOptimal };
	dsImgInfos[2] = { linearSampler, frameResources.normalImageView, vk::ImageLayout::eShaderReadOnlyOptimal };
	dsImgInfos[3] = { linearSampler, frameResources.metalRoughImageView, vk::ImageLayout::eShaderReadOnlyOptimal };
	dsImgInfos[4] = { linearSampler, frameResources.depthImageView, vk::ImageLayout::eShaderReadOnlyOptimal };
	std::vector<vk::WriteDescriptorSet> dsWrites(5);
	dsWrites[0] = { {}, 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &dsImgInfos[0] };
	dsWrites[1] = { {}, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &dsImgInfos[1] };
	dsWrites[2] = { {}, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &dsImgInfos[2] };
	dsWrites[3] = { {}, 3, 0, 1, vk::DescriptorType::eCombinedImageSampler, &dsImgInfos[3] };
	dsWrites[4] = { {}, 4, 0, 1, vk::DescriptorType::eCombinedImageSampler, &dsImgInfos[4] };
	frameResources.commBuff.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, renderPipelineLayout, 0, 5, dsWrites.data());

	frameResources.commBuff.draw(3, 1, 0, 0);

	frameResources.commBuff.endRendering();
}

void RenderEngine::changeImageLayoutForPresent(FrameResources& frameResources, uint32_t swapchainImageIndex)
{
	vk::ImageSubresourceRange imgRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
	vk::ImageMemoryBarrier2 barrier{ vk::PipelineStageFlagBits2::eColorAttachmentOutput, vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eBottomOfPipe, {}, vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR, graphicsQueueIdx, graphicsQueueIdx, swapchainImgs[swapchainImageIndex], imgRange };
	vk::DependencyInfo depInfo{ {}, 0, nullptr, 0, nullptr, 1, &barrier };

	frameResources.commBuff.pipelineBarrier2(depInfo);
}

FrameResources& RenderEngine::getNextFrameResources()
{
	frameIndex = (frameIndex + 1) % framesResources.size();
	return framesResources[frameIndex];
}

void RenderEngine::run()
{
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		draw();
	}
}

void RenderEngine::dispose()
{

}