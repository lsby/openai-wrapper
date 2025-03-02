import { GlobalAsyncItem, GlobalService } from '@lsby/ts-global'
import { Log } from '@lsby/ts-log'

export let Global = new GlobalService([
  new GlobalAsyncItem('log', async () => {
    return new Log('@lsby:openai-wrapper')
  }),
])
