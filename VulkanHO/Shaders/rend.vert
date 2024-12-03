#version 460

struct VertexData
{
    vec4 pos;
    vec2 uv;
};

VertexData vertexesData[3] =
{
    VertexData(vec4(-1, -1, 0, 1), vec2(0, 0)),
    VertexData(vec4(-1, 3, 0, 1), vec2(0, 2)),
    VertexData(vec4(3, -1, 0, 1), vec2(2, 0))
};

layout(location = 0) out vec2 uvOut;

void main()
{
    gl_Position = vertexesData[gl_VertexIndex].pos;
    uvOut = vertexesData[gl_VertexIndex].uv;
}