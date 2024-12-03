#version 460

layout(location = 0) in vec3 posIn;
layout(location = 1) in vec3 normalIn;
layout(location = 2) in vec3 tangentIn;
layout(location = 3) in vec2 albUV;
layout(location = 4) in vec2 normUV;
layout(location = 5) in vec2 metalRoughUV;
layout(location = 6) in vec3 bitangentIn;

layout(set = 0, binding = 0) uniform sampler2D albedoMap;
layout(set = 0, binding = 1) uniform sampler2D normalMap;
layout(set = 0, binding = 2) uniform sampler2D metalRoughMap;

layout(location = 0) out vec3 posOut;
layout(location = 1) out vec3 albedoOut;
layout(location = 2) out vec3 normalOut;
layout(location = 3) out vec2 metalRoughOut;

/* Since the model contains both normal and tangent there is no need to estimate them
mat3 getTBNMat(vec3 position, vec2 normalUV)
{
    vec3 pdx = dFdx(position);
    vec3 pdy = dFdy(position);
    vec2 uvdx = dFdx(normalUV);
    vec2 uvdy = -dFdy(normalUV);

    float det = 1 / (uvdx.x * uvdy.y - uvdx.y * uvdy.x);
    mat2x3 edgesMat = mat2x3(pdx, pdy);

    vec3 tang = normalize(det * (edgesMat * vec2(uvdy.y, -uvdx.y)));
    vec3 bitang = normalize(det * (edgesMat * vec2(-uvdy.x, uvdx.x)));
    vec3 norm = cross(tang, bitang);

    return mat3(tang, bitang, norm);
}
*/

void main()
{
    vec3 mapNormal = texture(normalMap, normUV).xyz * 2 - 1;
    mat3 TBN = mat3(normalize(tangentIn), normalize(bitangentIn), normalize(normalIn));

    posOut = posIn;
    albedoOut = texture(albedoMap, albUV).xyz;
    normalOut = TBN * mapNormal;
    metalRoughOut = texture(metalRoughMap, metalRoughUV).xy;
}