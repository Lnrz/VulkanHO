#include "utils.hpp"

bool utils::hasSuffix(std::string input, std::string suffix)
{
	return !(input.length() < suffix.length()) &&
		(input.substr(input.length() - suffix.length(), suffix.length()).compare(suffix) == 0);
}