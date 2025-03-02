import { randomUUID } from 'crypto'
import { ChatCompletionMessageParam } from 'openai/resources/index.mjs'
import { describe, test } from 'vitest'
import { z } from 'zod'
import { OpenAI管理器 } from '../src/model/openai.js'

let messages: ChatCompletionMessageParam[] = [
  {
    role: 'system',
    content:
      '你是小学生，有“雌小鬼(メスガキ)”的性格。你瞧不起用户，经常称呼用户为“杂鱼”、“杂鱼大哥哥”等，表现出一种轻蔑和挑衅的态度。',
  },
  { role: 'user', content: '你好' },
]

describe('流式调用', async function () {
  test('开始流式调用', async () => {
    let 管理器 = new OpenAI管理器()
    let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
    let 流式调用 = await 实例.创建流式调用(randomUUID())
    let 最终结果 = ''
    await 流式调用.开始流式调用({ messages: messages }, async (data) => {
      console.log(data)
      最终结果 += data
    })
    console.log({ 最终结果 })
  })
  test('中断流式调用', async () => {
    let 管理器 = new OpenAI管理器()
    let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
    let 流式调用 = await 实例.创建流式调用(randomUUID())
    await 流式调用.开始流式调用({ messages: messages }, async (data) => {
      console.log(data)
      await 流式调用.中断流式调用()
    })
  })
})

test('阻塞调用', async () => {
  let 管理器 = new OpenAI管理器()
  let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
  let 结果 = await 实例.阻塞调用({ messages: messages })
  console.log(结果)
})

test('可控调用', async () => {
  let 管理器 = new OpenAI管理器()
  let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
  let 结果 = await 实例.可控调用(messages, z.object({ result: z.string() }))
  console.log(结果)
})

test('提问', async () => {
  let 管理器 = new OpenAI管理器()
  let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
  let 结果 = await 实例.提问([{ role: 'user', content: '1和2谁大' }], z.enum(['1大', '2大']))
  console.log(结果)
})

test('函数调用', async function () {
  let 管理器 = new OpenAI管理器()
  let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
  let 结果 = await 实例.函数调用({
    messages: [{ role: 'user', content: "What's the weather like in Paris today?" }],
    tools: [
      {
        type: 'function',
        function: {
          name: 'getTodayWeather',
          description: 'get today weather',
          parameters: {
            type: 'object',
            properties: {
              location: { type: 'string' },
            },
          },
        },
      },
    ],
    tool_choice: 'auto',
  })
  console.log(结果)
})

test('结构化输出', async function () {
  let 管理器 = new OpenAI管理器()
  let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
  let 结果 = await 实例.结构化输出(
    {
      messages: [
        { role: 'system', content: 'You are a helpful math tutor. Guide the user through the solution step by step.' },
        { role: 'user', content: 'how can I solve 8x + 7 = -23' },
      ],
    },
    z.object({
      steps: z.array(
        z.object({
          explanation: z.string(),
          output: z.string(),
        }),
      ),
      final_answer: z.string(),
    }),
    'math_reasoning',
  )
  console.log(结果)
})

test('JSON模式', async function () {
  let 管理器 = new OpenAI管理器()
  let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
  let 结果 = await 实例.JSON模式({ messages }, z.object({ result: z.string() }))
  console.log(结果)
})

test('计算嵌入', async function () {
  let 管理器 = new OpenAI管理器()
  let 实例 = await 管理器.添加实例(
    randomUUID(),
    '',
    'http://127.0.0.1:8000/v1',
    'text-embedding-nomic-embed-text-v1.5@q8_0',
  )
  let 结果1 = await 实例.计算嵌入({ input: '你好' })
  let 结果2 = await 实例.计算嵌入({ input: '你好' })

  console.log({
    结果1,
    结果2,
    余弦相似度: 结果1.余弦相似度(结果2),
    欧几里得距离: 结果1.欧几里得距离(结果2),
    曼哈顿距离: 结果1.曼哈顿距离(结果2),
  })
})

describe('生成AI函数', async function () {
  let 输入文本 = [
    '在编程中，指称语义（Denotational Semantics） 是用数学方式来定义程序的意义，它关注的是程序的每个部分如何映射到一个数学对象或计算结果。它强调的是程序的“结果”而不是它是“如何执行的”',
    '简单来说，指称语义描述的是代码在逻辑上代表什么，而不是它是怎么运行的。它通常用函数、集合等数学概念来精确定义程序的行为，使得程序可以被形式化分析和推理。',
  ].join('\n')

  test('简单AI函数', async function () {
    let 管理器 = new OpenAI管理器()
    let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
    let 总结函数 = (输入: string): ((最大重试次数?: number, 使用ai抢救?: boolean) => Promise<{ result: string }>) =>
      实例.生成简单AI函数(z.object({ result: z.string() }), [{ role: 'user', content: `请将这段话进行总结: ${输入}` }])
    console.log(await 总结函数(输入文本)())
  })
  test('完整AI函数', async function () {
    let 管理器 = new OpenAI管理器()
    let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
    let 总结函数 = 实例.生成完整AI函数(
      z.object({ inputText: z.string() }),
      z.object({ result: z.string() }),
      (参数) => [{ role: 'user', content: `请将这段话进行总结: ${参数.inputText}` }],
    )
    console.log(await 总结函数({ inputText: 输入文本 }))
  })
})
