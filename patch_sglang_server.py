import sys
import os

filepath = '/root/atropos/atroposlib/envs/server_handling/sglang_server.py'

with open(filepath, 'r') as f:
    lines = f.readlines()

new_lines = []
imported_httpx = False
for line in lines:
    if not imported_httpx and 'import asyncio' in line:
        new_lines.append(line)
        new_lines.append('import httpx\n')
        imported_httpx = True
    else:
        new_lines.append(line)

content = ''.join(new_lines)

# Robust Health Check Implementation
robust_check_code = """
    async def check_server_status_task(self, chat_completion: bool = True):
        # Robust health check logic for SGLang native driver.
        # Order: /model_info -> / -> /generate
        while True:
            try:
                successful = False
                async with httpx.AsyncClient(timeout=2.0) as client:
                    # 1. try GET /model_info
                    try:
                        resp = await client.get(f'{self.config.base_url}/model_info')
                        if resp.status_code == 200:
                            successful = True
                    except Exception:
                        pass
                    
                    # 2. if fails -> try GET /
                    if not successful:
                        try:
                            resp = await client.get(f'{self.config.base_url}/')
                            if resp.status_code == 200:
                                successful = True
                        except Exception:
                            pass
                            
                    # 3. if fails -> try lightweight POST /generate
                    if not successful:
                        try:
                            payload = {
                                'text': 'hi',
                                'sampling_params': {'max_new_tokens': 1},
                            }
                            resp = await client.post(f'{self.config.base_url}/generate', json=payload)
                            if resp.status_code == 200:
                                successful = True
                        except Exception:
                            pass
                
                self.server_healthy = successful
            except Exception:
                self.server_healthy = False
                
            await asyncio.sleep(1)
"""

start_line = -1
end_line = -1
for i, line in enumerate(new_lines):
    if 'def check_server_status_task' in line:
        start_line = i
    if start_line != -1 and 'def _chat_completion_wrapper' in line:
        end_line = i
        break

if start_line != -1 and end_line != -1:
    content_replaced = ''.join(new_lines[:start_line]) + robust_check_code + '\n' + ''.join(new_lines[end_line:])
    with open(filepath, 'w') as f:
        f.write(content_replaced)
    print("Successfully patched sglang_server.py with robust health check.")
else:
    print(f"FAILED to find health check method indices. start={start_line}, end={end_line}")
