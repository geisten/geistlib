"""Self-authored completion smoke cases; not a standard quality benchmark.

Reuses the fixed tasks from the M6 precision experiment, with a longer
background prefix to exercise Bonsai native prefill at >=512 tokens.
"""
CONTEXT = 'Read the following short reference notes, then complete the final sentence. A library keeps books on numbered shelves and records each loan in a catalogue. Visitors can ask a librarian to find a title, renew a loan, or reserve a quiet room. The reading room opens in the morning and closes in the evening. A community garden beside the library has flowers, vegetables, and several fruit trees. Volunteers water the plants regularly, repair the paths, and share the harvest. Weather reports help them plan their work. They keep a notebook with the date, temperature, rainfall, and tasks completed each day. These notes provide background only; answer the final prompt using its own facts.\n\n' * 5

CASES = [('Paris', 'France is a country in western Europe. Its largest city and capital is', ' Paris, which is known for the Eiffel Tower and the river Seine.'), ('arithmetic', 'A shop has twelve boxes with six pencils in each box. The total number of pencils is', ' seventy-two, because twelve multiplied by six equals seventy-two.'), ('water', 'At standard atmospheric pressure, pure water freezes at zero degrees Celsius and boils at', ' one hundred degrees Celsius. Pressure changes the boiling point.'), ('German', 'Berlin ist die Hauptstadt von Deutschland. Die Hauptstadt von Frankreich ist', ' Paris. Beide Städte sind wichtige politische und kulturelle Zentren in Europa.'), ('Python', 'In Python, a function that returns the square of its argument can be written as:\ndef square(x):\n    return', ' x * x\n\nCalling square(5) returns 25.'), ('JSON', 'A valid JSON object uses double quotes around keys and string values. For example:\n{"name": "Ada", "age":', ' 36, "active": true}'), ('sorting', 'To sort the numbers 9, 2, 7, and 4 in ascending order, begin with the smallest. The sorted list is', ' 2, 4, 7, 9. Each number is greater than the one before it.'), ('biology', 'Plants use sunlight, water, and carbon dioxide to make sugars. This process is called', ' photosynthesis, and it releases oxygen into the atmosphere.'), ('reasoning', 'All squares are rectangles, but not all rectangles are squares. A square has four equal sides and', ' four right angles. A rectangle does not need to have four equal sides.'), ('translation', 'The French greeting "Bonjour" is commonly used during the day. In English, it means', ' "Hello" or "Good morning", depending on the context.'), ('history', 'The first crewed landing on the Moon took place in 1969 during the Apollo 11 mission. The first person to walk on the Moon was', ' Neil Armstrong, followed by Buzz Aldrin.'), ('context', 'Mira put a red book on the table and a blue book on the shelf. Later, she picked up the book from the table. The color of the book she picked up was', ' red, because the red book was on the table.')]

CASES = [(name, CONTEXT + prompt, continuation) for name,prompt,continuation in CASES]

# Text-only single user turn of the model's GGUF Qwen 3.5 template with
# enable_thinking=False and add_generation_prompt=True. No template engine
# or model download is needed for this fixed benchmark conversation.
CHAT_CASES = [(name,
               '<|im_start|>system\nComplete the final sentence. Output only its continuation, without repeating the prompt.<|im_end|>\n'
               '<|im_start|>user\n' + prompt.rstrip() + '<|im_end|>\n'
               '<|im_start|>assistant\n<think>\n\n</think>\n\n',
               continuation.lstrip()) for name, prompt, continuation in CASES]
