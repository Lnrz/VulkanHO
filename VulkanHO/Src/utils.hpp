#pragma

#include <vkb/VkBootstrap.h>
#include <iostream>
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace utils
{
	template<typename T>
	void checkVkbError(vkb::Result<T> res)
	{
		if (!res)
		{
			std::cout << res.full_error().type << std::endl;
			std::cout << res.error().message() << std::endl;
			exit(-1);
		}
	}

	template<typename T>
	glm::mat4 glmVkPerspective(T fovY, T aspect, T zNear, T zFar)
	{
		auto res = glm::perspective(fovY, aspect, zNear, zFar);
		res[1][1] *= -1;
		return res;
	}

	template<typename T>
	glm::mat4 glmVkPerspectiveFov(T fov, T width, T height, T zNear, T zFar)
	{
		auto res = glm::perspectiveFov(fov, width, height, zNear, zFar);
		res[1][1] *= -1;
		return res;
	}

	bool hasSuffix(std::string input, std::string suffix);
}