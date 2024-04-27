package huplay.demo.config;

import huplay.demo.transformer.BaseTransformer;
import huplay.demo.transformer.eleutherai.gptneo.GPTNeo;
import huplay.demo.transformer.huggingface.bloom.Bloom;
import huplay.demo.transformer.meta.llama.Llama;
import huplay.demo.transformer.openai.gpt1.GPT1;
import huplay.demo.transformer.openai.gpt2.GPT2;

public enum TransformerType
{
    OPENAI_GPT_1,
    OPENAI_GPT_2,
    HUGGING_FACE_BLOOM,
    ELEUTHERAI_NEO,
    META_LLAMA;

    public BaseTransformer getTransformer(Config config)
    {
        switch (this)
        {
            case OPENAI_GPT_1: return new GPT1(config);
            case OPENAI_GPT_2: return new GPT2(config);
            case HUGGING_FACE_BLOOM: return new Bloom(config);
            case ELEUTHERAI_NEO: return new GPTNeo(config);
            case META_LLAMA: return new Llama(config);
            default:
                throw new RuntimeException("Unknown transformer type: " + config.getTransformerType());
        }
    }
}
