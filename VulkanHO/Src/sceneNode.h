#pragma once

#include <glm/glm.hpp>
#include "surfaceData.h"
#include <memory>

struct SceneNode
{
	std::weak_ptr<SceneNode> parent{};
	std::vector<std::shared_ptr<SceneNode>> children{};
	std::vector<SurfaceData> surfacesDatas{};
	glm::mat4 localMatrix{ 1 };
};