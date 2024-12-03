#version 460

layout(location = 0) in vec2 uv;

layout(set = 0, binding = 0) uniform sampler2D posMap;
layout(set = 0, binding = 1) uniform sampler2D albedoMap;
layout(set = 0, binding = 2) uniform sampler2D normalMap;
layout(set = 0, binding = 3) uniform sampler2D metalRoughMap;
layout(set = 0, binding = 4) uniform sampler2D depthMap;

layout(location = 0) out vec4 col;

struct Light
{
    vec3 pos; // lights position wrt eye
    vec3 col;
};

const uint lightsNum = 2;
const Light lights[2] =
{
    Light(vec3(0.0, 0.3, -0.2), vec3(1.0)),
    Light(vec3(0.0, 0.0, -2.0), vec3(1.0))
};
const float lightWeight = 1.0 / lightsNum;

const float PI = 3.14159265359;

float  normalDistributionFactorGGXTR(float cosNormalHalfway, float roughness)
{
    float rr;
    float tmp;
    float res;

    rr = roughness * roughness;
    tmp = cosNormalHalfway * cosNormalHalfway * (rr - 1.0) + 1.0;
    tmp = tmp * tmp;
    res = rr / (PI * tmp);

    return res;
}

float geometryFunctionSchlickGGX(float cosNormalVector, float k)
{
    return cosNormalVector / (cosNormalVector * (1 - k) + k);
}

float geometryFunctionSchlickGGX(float cosNormalView, float cosNormalLight, float k)
{
    return geometryFunctionSchlickGGX(cosNormalView, k) * geometryFunctionSchlickGGX(cosNormalLight, k);
}

vec3 fresnelSchlickApproximation(float cosHalfwayView, vec3 F0)
{
    return F0 + (1 - F0) * pow(1 - cosHalfwayView, 5);
}

void main()
{
    vec3 pos = texture(posMap, uv).xyz;
    vec3 albedo = texture(albedoMap, uv).rgb;
    vec3 normal = texture(normalMap, uv).xyz;
    float roughness = texture(metalRoughMap, uv).x;
    float metallic = texture(metalRoughMap, uv).y;

    vec3 view = normalize(-pos);
    vec3 posLightVec;
    float posLightDist;
    vec3 light;
    vec3 halfway;
    float cosNLAngle;
    float cosNVAngle;
    float cosNHAngle;
    float cosVHAngle;
    float rollOffFactor;
    vec3 inRadiance;
    float D;
    float k;
    float G;
    vec3 F0;
    vec3 F;
    
    vec3 outRadiance = vec3(0.0);
    for (uint i = 0; i < lightsNum; i++)
    {
        posLightVec = lights[i].pos - pos;
        posLightDist = length(posLightVec);
        light = normalize(posLightVec);
        halfway = normalize(light + view);
        cosNLAngle = max(dot(normal, light), 0);
        cosNVAngle = max(dot(normal, view), 0);
        cosNHAngle = max(dot(normal, halfway), 0);
        cosVHAngle = max(dot(view, halfway), 0);
        rollOffFactor = 1.0 / posLightDist;
        inRadiance = lights[i].col * rollOffFactor;
        D = normalDistributionFactorGGXTR(cosNHAngle, roughness);
        k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
        G = geometryFunctionSchlickGGX(cosNVAngle, cosNLAngle, k);
        F0 = mix(vec3(0.04), albedo, metallic);
        F = fresnelSchlickApproximation(cosVHAngle, F0);
        outRadiance += ((1.0 - metallic) * (1.0 - F) * albedo / PI + (D * F * G) / (4.0 * cosNVAngle * cosNLAngle + 0.001)) * inRadiance * cosNLAngle * lightWeight;
    }

    vec3 ambient = vec3(0.05) * albedo;
    col = vec4(ambient + outRadiance, 1);
}