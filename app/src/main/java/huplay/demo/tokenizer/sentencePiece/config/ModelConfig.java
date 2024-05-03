package huplay.demo.tokenizer.sentencePiece.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.LinkedHashMap;

@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelConfig
{
    @JsonProperty
    public String type;

    @JsonProperty("unk_token")
    public String unknownToken;

    @JsonProperty("vocab")
    public LinkedHashMap<String, Float> vocabulary;

    @Override
    public String toString()
    {
        return "ModelConfig{" +
                "type='" + type + '\'' +
                ", unknownToken='" + unknownToken + '\'' +
                ", vocabulary=" + vocabulary +
                '}';
    }
}
