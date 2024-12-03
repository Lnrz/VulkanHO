#pragma once

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <vkb/VkBootstrap.h>
#include "frameResources.h"
#include "matrixData.h"
#include "sceneNode.h"

class RenderEngine
{
public:

	void init();
	void run();
	void dispose();

private:

	vk::Instance vkInstance;
	vk::DebugUtilsMessengerEXT vkDebugMess;
	vk::PhysicalDevice vkPhysicalDevice;
	vk::Device vkDevice;
	vk::SwapchainKHR vkSwapchain;
	vk::Extent2D swapchainImageExtent;
	vk::Format swapchainImageFormat;
	std::vector<vk::Image> swapchainImgs{};
	std::vector<vk::ImageView> swapchainImgsViews{};
	VmaAllocator allocator;
	vk::SurfaceKHR vkSurface;
	vk::Queue graphicsQueue;
	uint32_t graphicsQueueIdx;
	vk::CommandPool mainCommPool;
	std::array<FrameResources, 3> framesResources{};
	uint8_t frameIndex{ 0 };
	vk::Pipeline geometricPipeline;
	vk::DescriptorSetLayout geometricDescriptorSetLayout;
	vk::PipelineLayout geometricPipelineLayout;
	vk::Pipeline renderPipeline;
	vk::DescriptorSetLayout renderDescriptorSetLayout;
	vk::PipelineLayout renderPipelineLayout;
	vk::Sampler linearSampler;
	vk::Sampler nearestSampler;
	vk::DescriptorPool modelsDescriptorPool;
	std::vector<std::shared_ptr<SceneNode>> sceneNodes;
	int windowWidth{ 720 }, windowHeight{ 480 };
	GLFWwindow* window;

	void createWindow();
	void createVkInstance(vkb::Instance & instance);
	void pickVkPhysicalDevice(const vkb::Instance & instance, vkb::PhysicalDevice & physicalDevice);
	void createVkDevice(const vkb::PhysicalDevice& physicalDevice, vkb::Device & device);
	void initVkHppDispatcher();
	void getQueues(const vkb::Device & device);
	void createSwapchain(const vkb::Device& device);
	void createAllocator();
	void createSamplers();
	void createDescriptorPools();
	void createDescriptorSetLayouts();
	void createCommandPools();
	void createFramesResources();
	void createCommandBuffers();
	void createSyncStructures();
	void loadModels();
	void createPipelines();

	void createGeometricDescriptorLayout();
	void createRenderDescriptorLayout();

	void createGeometricPipeline();
	void createRenderPipeline();

	void changeImagesLayoutForGeometric(FrameResources& frameResources);
	void changeImagesLayoutForRender(FrameResources& frameResources, uint32_t swapchainImageIndex);
	void changeImageLayoutForPresent(FrameResources& frameResources, uint32_t swapchainImageIndex);

	void draw();
	void drawGeometric(FrameResources& frameResources);
	void drawRender(FrameResources& frameResources, uint32_t swapchainImageIndex);

	FrameResources& getNextFrameResources();
};