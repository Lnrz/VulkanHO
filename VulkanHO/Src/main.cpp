#include "renderEngine.h"

int main(int argc, char* argv[])
{
	RenderEngine re{};

	re.init();
	re.run();
	re.dispose();

	return 0;
}