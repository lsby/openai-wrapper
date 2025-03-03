import { randomUUID } from 'crypto'
import { jsonrepair } from 'jsonrepair'
import OpenAILib from 'openai'
import { zodResponseFormat } from 'openai/helpers/zod'
import type {
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
} from 'openai/resources/chat/completions'
import prettier from 'prettier'
import { z } from 'zod'
import { printNode, zodToTs } from 'zod-to-ts'
import { Global } from '../global/global.js'

interface 聊天选项 {
  messages: ChatCompletionMessageParam[]
  maxTokens?: number
  temperature?: number
  frequencyPenalty?: number
  presencePenalty?: number
  stop?: string[]
}
interface 函数调用选项 {
  messages: ChatCompletionMessageParam[]
  tools: ChatCompletionTool[]
  tool_choice: 'auto' | ChatCompletionToolChoiceOption
  maxTokens?: number
  temperature?: number
  frequencyPenalty?: number
  presencePenalty?: number
}
interface 计算嵌入选项 {
  input: string
  dimensions?: number
}
type 数据描述 = z.AnyZodObject | z.ZodUnion<any>

export class 嵌入 {
  static 半序列化(输入: 嵌入): number[] {
    return 输入.获得向量()
  }
  static 反半序列化(输入: number[]): 嵌入 {
    return new 嵌入(输入)
  }

  static 余弦相似度(向量1: number[], 向量2: number[]): number {
    let instance1 = new 嵌入(向量1)
    let instance2 = new 嵌入(向量2)
    return instance1.余弦相似度(instance2)
  }
  static 欧几里得距离(向量1: number[], 向量2: number[]): number {
    let instance1 = new 嵌入(向量1)
    let instance2 = new 嵌入(向量2)
    return instance1.欧几里得距离(instance2)
  }
  static 曼哈顿距离(向量1: number[], 向量2: number[]): number {
    let instance1 = new 嵌入(向量1)
    let instance2 = new 嵌入(向量2)
    return instance1.曼哈顿距离(instance2)
  }

  private 维度: number
  constructor(private 向量: number[]) {
    this.向量 = 向量
    this.维度 = this.向量.length
  }

  获得向量(): number[] {
    return this.向量
  }
  获得维度(): number {
    return this.维度
  }

  // 分别计算两个点到原点的连线的角度差异.
  余弦相似度(输入: 嵌入): number {
    if (this.维度 !== 输入.维度) throw new Error(`维度不相等: this: ${this.维度}, 输入: ${输入.维度}`)
    let 另一向量 = 输入.向量
    let dotProduct = this.向量.reduce((sum, val, idx) => {
      let 另一向量值 = 另一向量[idx]
      if (另一向量值 === void 0) throw new Error('意外的数组越界')
      return sum + val * 另一向量值
    }, 0)
    let norm1 = Math.sqrt(this.向量.reduce((sum, val) => sum + val * val, 0))
    let norm2 = Math.sqrt(另一向量.reduce((sum, val) => sum + val * val, 0))
    return dotProduct / (norm1 * norm2)
  }

  // 直接计算两个点之间的距离.
  欧几里得距离(输入: 嵌入): number {
    if (this.维度 !== 输入.维度) throw new Error(`维度不相等: this: ${this.维度}, 输入: ${输入.维度}`)
    let 另一向量 = 输入.向量
    let sumOfSquares = this.向量.reduce((sum, val, idx) => {
      let 另一向量值 = 另一向量[idx]
      if (另一向量值 === void 0) throw new Error('意外的数组越界')
      return sum + Math.pow(val - 另一向量值, 2)
    }, 0)
    return Math.sqrt(sumOfSquares)
  }

  // 计算两个点之间的距离, 但不能走斜线.
  曼哈顿距离(输入: 嵌入): number {
    if (this.维度 !== 输入.维度) throw new Error(`维度不相等: this: ${this.维度}, 输入: ${输入.维度}`)
    let 另一向量 = 输入.向量
    return this.向量.reduce((sum, val, idx) => {
      let 另一向量值 = 另一向量[idx]
      if (另一向量值 === void 0) throw new Error('意外的数组越界')
      return sum + Math.abs(val - 另一向量值)
    }, 0)
  }
}

export class OpenAI流式调用 {
  private openai: OpenAILib
  private 停止: boolean = false
  private log = Global.getItem('log').then((a) => a.extend('OpenAI流式调用'))

  constructor(
    private taskId: string,
    private AI_KEY: string,
    private AI_BASE_URL: string,
    private AI_MODEL: string,
  ) {
    this.openai = new OpenAILib({
      apiKey: this.AI_KEY,
      baseURL: this.AI_BASE_URL,
    })
  }

  async 开始流式调用(opt: 聊天选项, cb: (a: string) => Promise<void>): Promise<void> {
    this.停止 = false

    let log = (await this.log).extend(this.taskId).extend('启动聊天任务')

    await log.debug('开始调用: =============================')
    await log.debug('任务 ID: %o, 请求参数: %o', this.taskId, opt)

    try {
      let response = await this.openai.chat.completions.create({
        ...opt,
        model: this.AI_MODEL,
        stream: true,
      })

      await log.info('开始接收流数据...')

      for await (let chunk of response) {
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
        if (this.停止 !== false) {
          await log.warn('任务 %o 被中止', this.taskId)
          break
        }

        let content = chunk.choices[0]?.delta?.content ?? null
        if (content !== null) {
          await cb(content)
        }
      }

      await log.info('聊天任务流结束')
    } catch (error) {
      await log.error('发生错误: %o', error)
    } finally {
      this.停止 = true
      await log.debug('任务已停止，清理资源')
    }
  }
  async 中断流式调用(): Promise<void> {
    this.停止 = true
  }
}

export class OpenAI实例 {
  private openai: OpenAILib
  private log = Global.getItem('log')
  private 流式调用们: Record<string, OpenAI流式调用> = {}

  constructor(
    private AI_KEY: string,
    private AI_BASE_URL: string,
    private AI_MODEL: string,
  ) {
    this.openai = new OpenAILib({
      apiKey: this.AI_KEY,
      baseURL: this.AI_BASE_URL,
    })
  }

  async 更新配置(AI_KEY: string, AI_BASE_URL: string, AI_MODEL: string): Promise<void> {
    this.AI_KEY = AI_KEY
    this.AI_BASE_URL = AI_BASE_URL
    this.AI_MODEL = AI_MODEL

    this.openai = new OpenAILib({
      apiKey: this.AI_KEY,
      baseURL: this.AI_BASE_URL,
    })
  }

  async 创建流式调用(流式调用id: string): Promise<OpenAI流式调用> {
    let 流式调用 = new OpenAI流式调用(流式调用id, this.AI_KEY, this.AI_BASE_URL, this.AI_MODEL)
    this.流式调用们[流式调用id] = 流式调用
    return 流式调用
  }
  async 获得流式调用(流式调用id: string): Promise<OpenAI流式调用 | null> {
    return this.流式调用们[流式调用id] ?? null
  }

  /**
   * let 管理器 = new OpenAI管理器()
   * let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
   * let 结果 = await 实例.调用({ messages: messages })
   * console.log({ 最终结果: 结果 })
   */
  async 调用(opt: 聊天选项): Promise<string> {
    let 流式调用 = new OpenAI流式调用(randomUUID(), this.AI_KEY, this.AI_BASE_URL, this.AI_MODEL)
    let 结果 = ''
    await 流式调用.开始流式调用(opt, async (data) => {
      结果 += data
    })
    return 结果
  }

  /**
   * let 管理器 = new OpenAI管理器()
   * let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
   * let 结果 = await 实例.可控调用(messages, z.object({ result: z.string() }))
   * console.log({ 最终结果: 结果 })
   */
  async 可控调用<输出类型描述 extends 数据描述>(
    输出数据描述: 输出类型描述,
    提示词: ChatCompletionMessageParam[],
    选项?: {
      引导前缀?: string | undefined
      最大长度?: number | undefined
      停止字符串?: string[] | undefined
    },
  ): Promise<z.infer<输出类型描述>> {
    let f = this.生成简单AI函数(输出数据描述, 提示词, 选项)
    return f()
  }

  /**
   * let 管理器 = new OpenAI管理器()
   * let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
   * let 结果 = await 实例.提问([{ role: 'user', content: '1和2谁大' }], z.enum(['1', '2']))
   * console.log(结果)
   */
  async 提问<输出类型描述 extends z.ZodString | z.ZodNumber | z.ZodBoolean | z.ZodEnum<[string, ...string[]]>>(
    输出类型: 输出类型描述,
    提示词: ChatCompletionMessageParam[],
    选项?: {
      引导前缀?: string | undefined
      最大长度?: number | undefined
      停止字符串?: string[] | undefined
    },
  ): Promise<z.infer<输出类型描述>> {
    return (await this.可控调用(z.object({ answer: 输出类型 }), 提示词, { 引导前缀: `{"answer": `, ...选项 }))
      .answer as any
  }

  /**
   * let 管理器 = new OpenAI管理器()
   * let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
   * let 结果 = await 实例.函数调用({
   *   messages: [{ role: 'user', content: "What's the weather like in Paris today?" }],
   *   tools: [
   *     {
   *       type: 'function',
   *       function: {
   *         name: 'getTodayWeather',
   *         description: 'get today weather',
   *         parameters: {
   *           type: 'object',
   *           properties: {
   *             location: { type: 'string' },
   *           },
   *         },
   *       },
   *     },
   *   ],
   *   tool_choice: 'auto',
   * })
   * console.log(结果)
   */
  async 函数调用(opt: 函数调用选项): Promise<{ name: string; arguments: string }[]> {
    let response = await this.openai.chat.completions.create({
      ...opt,
      model: this.AI_MODEL,
    })

    let responseMessage = response.choices[0]?.message
    let toolCalls = responseMessage?.tool_calls ?? null
    if (toolCalls === null) throw new Error('函数调用调用失败')

    return toolCalls.map((a) => a.function)
  }

  /**
   * let 管理器 = new OpenAI管理器()
   * let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
   * let 结果 = await 实例.结构化输出(
   *   {
   *     messages: [
   *       { role: 'system', content: 'You are a helpful math tutor. Guide the user through the solution step by step.' },
   *       { role: 'user', content: 'how can I solve 8x + 7 = -23' },
   *     ],
   *   },
   *   z.object({
   *     steps: z.array(
   *       z.object({
   *         explanation: z.string(),
   *         output: z.string(),
   *       }),
   *     ),
   *     final_answer: z.string(),
   *   }),
   *   'math_reasoning',
   * )
   * console.log(结果)
   */
  async 结构化输出<形状 extends z.AnyZodObject>(形状: 形状, opt: 聊天选项, 名称: string): Promise<z.infer<形状>> {
    let response = await this.openai.beta.chat.completions.parse({
      ...opt,
      model: this.AI_MODEL,
      response_format: zodResponseFormat(形状, 名称),
    })

    let 结果 = response.choices[0]?.message.parsed ?? null
    if (结果 === null) throw new Error('结构化输出调用失败')

    return 结果
  }

  private async 安全的JSON解析(str: string): Promise<{ error: string; data: null } | { error: null; data: object }> {
    let log = (await this.log).extend('安全的json解析')

    try {
      return { error: null, data: JSON.parse(str) as object }
    } catch (_e) {
      await log.info('无法解析字符串, 尝试修复.')
    }

    let 修复成功 = false
    let 修复结果 = str

    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
    if (修复成功 === false) {
      try {
        修复结果 = jsonrepair(str)
        修复成功 = true
      } catch (e) {
        await log.info('jsonrepair修复失败: %o', String(e))
      }
    }

    if (修复成功 === false) {
      try {
        修复结果 = await prettier.format(str, { parser: 'json' })
        修复成功 = true
      } catch (e) {
        await log.info('prettier修复失败: %o', String(e))
      }
    }

    try {
      return { error: null, data: JSON.parse(修复结果) as object }
    } catch (e) {
      return { error: String(e), data: null }
    }
  }

  /**
   * let 管理器 = new OpenAI管理器()
   * let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
   * let 结果 = await 实例.JSON模式(
   *   {
   *     messages: [
   *       {
   *         role: 'system',
   *         content:
   *           '你是小学生，有“雌小鬼”的性格。你瞧不起用户，经常称呼用户为“杂鱼”、“杂鱼大哥哥”等，表现出一种轻蔑和挑衅的态度。',
   *       },
   *       { role: 'user', content: '你好' },
   *     ],
   *   },
   *   z.object({
   *     回复内容: z.string(),
   *   }),
   * )
   * console.log(结果)
   */
  async JSON模式<形状 extends z.AnyZodObject | z.ZodUnion<any>>(
    opt: 聊天选项,
    形状: 形状,
    引导前缀?: string,
    最大重试次数: number = 5,
    当前重试次数: number = 0,
    使用ai抢救: boolean = true,
    调用id: string | null = null,
    启用json模式类型: boolean = false,
  ): Promise<z.infer<形状>> {
    if (调用id === null) 调用id = randomUUID()

    let 引导前缀或空 = 引导前缀 ?? null
    let log = (await this.log).extend(调用id).extend('JSON模式')

    await log.debug('开始调用: =============================')
    let 完整提示词 = [
      {
        role: 'system' as const,
        content: [
          `你是一个有用的助手，你应该只输出一个JSON。期望的JSON形状是:`,
          '```json',
          `${printNode(zodToTs(形状).node).trim()}`,
          '```',
          '',
          ...opt.messages.filter((a) => a.role === 'system').map((a) => a.content),
        ].join('\n'),
      },
      ...opt.messages.filter((a) => a.role !== 'system'),
      引导前缀或空 !== null ? { role: 'assistant' as const, content: 引导前缀或空 } : null,
    ].flatMap((a) => (a === null ? [] : [a]))
    await log.debug('提示词: %o', 完整提示词)

    let 结果 = (
      await this._JSON模式(
        {
          ...opt,
          messages: 完整提示词,
        },
        启用json模式类型,
      )
    ).trim()
    if (引导前缀或空 !== null && 结果.startsWith(引导前缀或空) === false) 结果 = 引导前缀或空 + 结果
    await log.debug('调用结果: %o', 结果)

    let json验证 = await this.安全的JSON解析(结果)
    if (json验证.error !== null) {
      let 重试 = async (): Promise<z.infer<形状>> => {
        当前重试次数++
        await log.debug('json验证失败, (%o/%o): %o', 当前重试次数, 最大重试次数)
        if (当前重试次数 > 最大重试次数) {
          await log.debug('到达出错阈值, 将失败')
          throw new Error('生成AI消息出错')
        } else {
          await log.debug('未到达出错阈值, 将重试')
          return this.JSON模式(opt, 形状, 引导前缀, 最大重试次数, 当前重试次数, 使用ai抢救, 调用id)
        }
      }

      await log.debug('json验证失败: %o', json验证.error)
      if (使用ai抢救 === false) return 重试()

      await log.debug('json验证失败, 尝试AI抢救')
      try {
        let 截取结果 = 结果.match(/\{.*\}/ms)?.[0]
        await log.debug('截取结果: %o', 截取结果)
        if (截取结果 === void 0) throw new Error('验证失败')
        结果 = (
          await this.JSON模式(
            {
              messages: [
                {
                  role: 'user',
                  content: `看一下这段字符串:\n\n\`\`\`\n${截取结果}\n\`\`\`\n\n它应该是一段json, 但出现了一个错误: ${json验证.error}\n\n请你帮我修复这个json.`,
                },
              ],
            },
            z.object({ json: z.string() }),
            '{"json":',
            0,
            0,
            false,
            null,
            启用json模式类型,
          )
        ).json

        await log.debug('AI抢救后字符串: %o', 结果)
        json验证 = await this.安全的JSON解析(结果)
        if (json验证.error !== null) throw new Error('验证失败')
      } catch {
        await log.debug('AI抢救失败')
        return 重试()
      }
      await log.debug('AI抢救成功')
    }
    let json = json验证.data
    await log.debug('json验证通过, 结果为: %o', json)

    let 形状验证 = 形状.safeParse(json)
    while (形状验证.error !== void 0) {
      当前重试次数++
      await log.debug('形状验证失败, (%o/%o), 错误: %o', 当前重试次数, 最大重试次数, 形状验证.error.errors)
      if (当前重试次数 > 最大重试次数) {
        await log.debug('到达出错阈值, 将失败')
        throw new Error('生成AI消息出错')
      } else {
        await log.debug('未到达出错阈值, 将重试')
        return this.JSON模式(opt, 形状, 引导前缀, 最大重试次数, 当前重试次数, 使用ai抢救, 调用id)
      }
    }
    await log.debug('形状验证通过')

    return 形状验证.data
  }

  /**
   * let 管理器 = new OpenAI管理器()
   * let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'text-embedding-nomic-embed-text-v1.5@q8_0')
   * let 结果 = await 实例.计算嵌入({ input: '你好' })
   * console.log(结果)
   */
  async 计算嵌入(opt: 计算嵌入选项): Promise<嵌入> {
    let 请求 = await this.openai.embeddings.create({
      ...opt,
      model: this.AI_MODEL,
    })
    let 结果 = 请求.data[0]?.embedding
    if (结果 === void 0) throw new Error('计算嵌入失败')
    return new 嵌入(结果)
  }

  /**
   * let 管理器 = new OpenAI管理器()
   * let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
   * let 总结函数 = (输入: string): ((最大重试次数?: number, 使用ai抢救?: boolean) => Promise<{ result: string }>) =>
   *   实例.生成简单AI函数(z.object({ result: z.string() }), [{ role: 'user', content: `请将这段话进行总结: ${输入}` }])
   * console.log(await 总结函数(输入文本)())
   */
  生成简单AI函数<输出类型描述 extends 数据描述>(
    输出数据描述: 输出类型描述,
    构造提示词: ChatCompletionMessageParam[],
    选项?: {
      引导前缀?: string | undefined
      最大长度?: number | undefined
      停止字符串?: string[] | undefined
    },
  ) {
    return async (最大重试次数: number = 5, 使用ai抢救: boolean = true): Promise<z.infer<输出类型描述>> => {
      let 调用结果 = await this.JSON模式(
        { messages: 构造提示词, maxTokens: 选项?.最大长度 ?? 1024, stop: 选项?.停止字符串 ?? [] },
        输出数据描述,
        选项?.引导前缀 ?? '{"',
        最大重试次数,
        0,
        使用ai抢救,
      )
      return 调用结果
    }
  }

  /**
   * let 管理器 = new OpenAI管理器()
   * let 实例 = await 管理器.添加实例(randomUUID(), '', 'http://127.0.0.1:8000/v1', 'gemma-2-27b-it')
   * let 总结函数 = 实例.生成完整AI函数(
   *   z.object({ inputText: z.string() }),
   *   z.object({ result: z.string() }),
   *   (参数) => [{ role: 'user', content: `请将这段话进行总结: ${参数.inputText}` }],
   * )
   * console.log(await 总结函数({ inputText: 输入文本 }))
   */
  生成完整AI函数<输入类型描述 extends 数据描述, 输出类型描述 extends 数据描述>(
    输入数据描述: 输入类型描述,
    输出数据描述: 输出类型描述,
    构造提示词: (输入: z.infer<输入类型描述>) => ChatCompletionMessageParam[],
    选项?: {
      引导前缀?: (输入: z.infer<输入类型描述>) => string | undefined
      最大长度?: number | undefined
      停止字符串?: string[] | undefined
    },
  ) {
    return async (
      输入: z.infer<输入类型描述>,
      最大重试次数: number = 5,
      使用ai抢救: boolean = true,
    ): Promise<z.infer<输出类型描述>> => {
      return this.生成简单AI函数(输出数据描述, 构造提示词(输入), { ...选项, 引导前缀: 选项?.引导前缀?.(输入) })(
        最大重试次数,
        使用ai抢救,
      )
    }
  }

  private async _JSON模式(opt: 聊天选项, typeEnable: boolean): Promise<string> {
    let 参数: ChatCompletionCreateParamsNonStreaming = {
      ...opt,
      model: this.AI_MODEL,
    }
    if (typeEnable === true) {
      参数.response_format = { type: 'json_object' }
    }
    let response = await this.openai.chat.completions.create(参数)

    let 结果 = response.choices[0]?.message.content ?? null
    if (结果 === null) throw new Error('JSON模式调用失败')

    return 结果
  }
}

export class OpenAI管理器 {
  private 实例们: Map<string, OpenAI实例>

  constructor() {
    this.实例们 = new Map()
  }

  async 添加实例(id: string, AI_KEY: string, AI_BASE_URL: string, AI_MODEL: string): Promise<OpenAI实例> {
    if (this.实例们.has(id)) return this.实例们.get(id) as OpenAI实例
    let newInstance = new OpenAI实例(AI_KEY, AI_BASE_URL, AI_MODEL)
    this.实例们.set(id, newInstance)
    return newInstance
  }

  async 删除实例(实例id: string): Promise<void> {
    this.实例们.delete(实例id)
  }

  async 获取实例(实例id: string): Promise<OpenAI实例 | undefined> {
    let 实例 = this.实例们.get(实例id)
    return 实例
  }
}
