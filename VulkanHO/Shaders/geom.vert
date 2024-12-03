#version 460

layout(location = 0) in vec3 posIn;
layout(location = 1) in vec3 normalIn;
layout(location = 2) in vec4 tangentIn;
layout(location = 3) in vec2 albUVin;
layout(location = 4) in vec2 normUVin;
layout(location = 5) in vec2 metalRoughUVin;

layout(push_constant) uniform MatData
{
    mat4 mv;
    mat4 p;
} matrixes;

layout(location = 0) out vec3 posOut;
layout(location = 1) out vec3 normalOut;
layout(location = 2) out vec3 tangentOut;
layout(location = 3) out vec2 albUVout;
layout(location = 4) out vec2 normUVout;
layout(location = 5) out vec2 metalRoughUVout;
layout(location = 6) out vec3 bitangentOut;

void main()
{
    vec4 pos = matrixes.mv * vec4(posIn, 1);
    vec3 normal = (matrixes.mv * vec4(normalIn, 0)).xyz;
    vec3 tangent = (matrixes.mv * vec4(tangentIn.xyz, 0)).xyz;
    vec3 bitangent = cross(normal, tangent) * tangentIn.w;

    posOut = pos.xyz;
    normalOut = normal;
    tangentOut = tangent;
    bitangentOut = bitangent;
    albUVout = albUVin;
    normUVout = normUVin;
    metalRoughUVout = metalRoughUVin;

    gl_Position = matrixes.p * pos;
}