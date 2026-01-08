# @lsby/openai-wrapper

一个功能完整的TypeScript库，为OpenAI API提供便捷的高级接口。支持流式调用、结构化输出、函数调用、向量嵌入等功能，内置智能JSON修复和AI智能修复机制。

## 特性

- 📡 **多种调用方式**：流式调用、阻塞调用、可控调用
- 🎯 **结构化输出**：集成Zod，提供类型安全的API返回值
- 🔧 **函数调用**：支持OpenAI的function calling功能
- 💾 **向量嵌入**：内置向量操作，支持余弦相似度、欧几里得距离等计算
- 🛡️ **智能修复**：自动修复malformed JSON，支持AI辅助修复
- 🔄 **自动重试**：失败自动重试，可配置最大重试次数
- ⚡ **TypeScript支持**：完整的类型定义，支持CJS和ESM

## 安装

```bash
# npm
npm install @lsby/openai-wrapper

# pnpm
pnpm add @lsby/openai-wrapper

# yarn
yarn add @lsby/openai-wrapper
```

## 快速开始

### 基础用法

```typescript
import { OpenAI管理器 } from '@lsby/openai-wrapper'
import { randomUUID } from 'crypto'

// 创建管理器
const 管理器 = new OpenAI管理器()

// 添加OpenAI实例
const 实例 = await 管理器.添加实例(randomUUID(), 'your-api-key', 'https://api.openai.com/v1', 'gpt-4')

// 简单调用
const 结果 = await 实例.调用({
  messages: [
    { role: 'system', content: '你是一个有用的助手' },
    { role: 'user', content: '你好' },
  ],
})

console.log(结果) // AI的回复
```

### 流式调用

```typescript
const 流式调用 = await 实例.创建流式调用(randomUUID())

let 完整输出 = ''
await 流式调用.开始流式调用(
  {
    messages: [{ role: 'user', content: '写一个故事' }],
  },
  async (chunk) => {
    console.log(chunk) // 实时输出每个token
    完整输出 += chunk
  },
)

// 中断流式调用
await 流式调用.中断流式调用()
```

### 结构化输出

```typescript
import { z } from 'zod'

// 定义期望的输出结构
const 形状 = z.object({
  steps: z.array(
    z.object({
      explanation: z.string(),
      output: z.string(),
    }),
  ),
  final_answer: z.string(),
})

const 结果 = await 实例.结构化输出(
  形状,
  {
    messages: [
      { role: 'system', content: '你是一个数学老师' },
      { role: 'user', content: '如何解这个方程: 8x + 7 = -23' },
    ],
  },
  'math_reasoning',
)

console.log(结果) // { steps: [...], final_answer: '...' }
```

### 可控调用（推荐）

```typescript
import { z } from 'zod'

const 形状 = z.object({
  result: z.string(),
})

// 自动解析和验证返回值
const { result } = await 实例.可控调用(形状, [{ role: 'user', content: '简化一下这个问题' }])

console.log(result)
```

### 提问（简化版）

```typescript
import { z } from 'zod'

// 快速提问，自动包装为JSON格式
const 答案 = await 实例.提问(z.enum(['选项A', '选项B', '选项C']), [{ role: 'user', content: '哪个更好？' }])

console.log(答案) // '选项A'
```

### 函数调用

```typescript
const 工具 = [
  {
    type: 'function' as const,
    function: {
      name: 'getTodayWeather',
      description: '获取今天的天气',
      parameters: {
        type: 'object',
        properties: {
          location: { type: 'string', description: '城市名称' },
        },
        required: ['location'],
      },
    },
  },
]

const 结果 = await 实例.函数调用({
  messages: [{ role: 'user', content: '北京今天天气怎么样？' }],
  tools: 工具,
  tool_choice: 'auto',
})

// 返回 [{ name: 'getTodayWeather', arguments: '{\"location\": \"北京\"}' }]
```

### JSON 模式

```typescript
import { z } from 'zod'

// 定义期望的JSON结构
const 用户信息 = z.object({
  name: z.string(),
  age: z.number(),
  email: z.string().email(),
  tags: z.array(z.string()),
})

// 使用JSON模式，内置错误修复和AI智能修复
const 结果 = await 实例.JSON模式(
  用户信息,
  {
    messages: [
      { role: 'user', content: '生成一个用户信息，名字叫张三，30岁，邮箱是zhangsan@example.com，标签有开发者和设计师' },
    ],
  },
)

console.log(结果) // { name: '张三', age: 30, email: 'zhangsan@example.com', tags: ['开发者', '设计师'] }

// 支持自定义参数
const 结果2 = await 实例.JSON模式(
  用户信息,
  { messages: [...] },
  '{"name": "',  // 引导前缀
  5,             // 最大重试次数
  0,             // 当前重试次数（一般不需要手动设置）
  true,          // 是否启用AI智能修复机制
)
```

### 向量嵌入

```typescript
import { 嵌入 } from '@lsby/openai-wrapper'

// 计算嵌入
const 嵌入对象 = await 实例.计算嵌入({
  input: '你好世界',
})

const 向量 = 嵌入对象.获得向量() // number[]
const 维度 = 嵌入对象.获得维度() // 1536

// 计算相似度
const 嵌入2 = await 实例.计算嵌入({ input: '你好' })
const 相似度 = 嵌入对象.余弦相似度(嵌入2) // -1 to 1

// 计算距离
const 欧距 = 嵌入对象.欧几里得距离(嵌入2)
const 曼距 = 嵌入对象.曼哈顿距离(嵌入2)
```

## 智能修复机制

该库内置了强大的智能修复机制，确保JSON解析的稳定性。

### 工作流程

1. **初始解析**：尝试直接解析AI返回的JSON
2. **自动修复**：如果解析失败，按顺序尝试：
   - 使用 `jsonrepair` 库修复格式错误的JSON
   - 使用 `prettier` 格式化JSON
3. **AI智能修复**（当启用时）：如果自动修复失败
   - 提取JSON对象部分（使用正则 `/\{.*\}/ms`）
   - 调用AI进行修复，请求AI识别并修复JSON中的错误
   - 重新验证修复后的JSON
4. **重试机制**：验证失败后自动重试，直到达到最大重试次数

### 使用示例

```typescript
// JSON模式默认启用AI智能修复机制
const 结果 = await 实例.JSON模式(
  z.object({ answer: z.string() }),
  { messages: [{ role: 'user', content: '你是谁' }] },
  undefined,  // 引导前缀
  5,          // 最大重试次数
  0,          // 当前重试次数
  true,       // 启用AI智能修复机制 ← 默认为true
)

// 禁用AI智能修复机制（仅使用自动修复）
const 结果2 = await 实例.JSON模式(
  z.object({ answer: z.string() }),
  { messages: [...] },
  undefined,
  5,
  0,
  false,  // 禁用AI智能修复
)
```

### 机制详解

**简单来说，智能修复机制就是一个自我修复的过程：**

假设AI返回了不完整的JSON：

```
{"name": "张三", "age: 30
```

处理流程：

1. ❌ 直接解析失败（JSON不完整）
2. 🔧 自动工具修复失败（格式破损过大）
3. 🆘 启动AI智能修复：把残破的JSON发给AI，说"帮我修复这个JSON"
4. ✅ AI修复后返回完整的JSON：`{"name": "张三", "age": 30}`
5. ✨ 验证通过，任务完成

**适用场景：**

- AI返回的JSON被截断了
- JSON里混入了额外的文字
- 括号或逗号不匹配
- 字段缺失或格式错误
