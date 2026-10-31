from mcp import Client, StdioServerParameters
import sys
import asyncio


server_parameters = StdioServerParameters(
    command=sys.executable,
    args=["09_mcp/mcp_server_basic.py"]
)


async def main():
    async with Client(server_parameters) as client:
        tool_result = await client.list_tools()
        print(tool_result.tools)

        result = await client.call_tool(
            "add",
            {
                "a": 2,
                "b": 3
            }
        )

        final_result = result.structured_content["result"]
        print(final_result)


if __name__ == "__main__":
    asyncio.run(main())
