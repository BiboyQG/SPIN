import openai

response = openai.ChatCompletion.create(
        model="Qwen/Qwen1.5-32B-Chat-AWQ",
        messages=[
            {"role": "system", "content": "You are an expert at summarizing car review articles in JSON format."},
            {"role": "user", "content": "hello"}
        ],
        max_tokens=14000,
        temperature=0.8
    )

print(response.choices[0].message.content)