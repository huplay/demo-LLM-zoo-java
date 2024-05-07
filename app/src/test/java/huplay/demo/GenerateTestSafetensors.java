package huplay.demo;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.demo.config.ParameterReader;
import huplay.demo.config.SafetensorsModel;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

import static huplay.demo.config.SafetensorsModel.TensorModel;

/**
 * Generates test .safetensors file to use in unit tests
 * It uses a subset of the values of a .safetensors file (not random, to make the scale real)
 * (Supports only 1 or 2 dimensions, float32 values, and uses a fixed metadata)
 *
 * @author Hunor Szegi
 */
public class GenerateTestSafetensors
{
    public static void main(String... args) throws IOException
    {
        LinkedHashMap<String, int[]> testEntries = getTestEntriesGPT1();
        //LinkedHashMap<String, int[]> testEntries = getTestEntriesGPT2();
        //LinkedHashMap<String, int[]> testEntries = getTestEntriesGPTNEO();

        generate("d:/test", "model.safetensors", "test.safetensors", testEntries);
    }

    private static LinkedHashMap<String, int[]> getTestEntriesGPT1()
    {
        int tokenCount = 10;
        int hiddenSize = 12;
        int contextSize = 10;
        int feedForwardSize = hiddenSize * 4;

        LinkedHashMap<String, int[]> testEntries = new LinkedHashMap<>();

        testEntries.put("tokens_embed.weight", new int[] {tokenCount, hiddenSize});
        testEntries.put("positions_embed.weight", new int[] {contextSize, hiddenSize});

        testEntries.put("h.0.ln_1.weight", new int[] {hiddenSize});
        testEntries.put("h.0.ln_1.bias", new int[] {hiddenSize});
        testEntries.put("h.0.attn.c_attn.weight", new int[] {hiddenSize, hiddenSize * 3});
        testEntries.put("h.0.attn.c_attn.bias", new int[] {hiddenSize * 3});
        testEntries.put("h.0.attn.c_proj.weight", new int[] {hiddenSize, hiddenSize});
        testEntries.put("h.0.attn.c_proj.bias", new int[] {hiddenSize});
        testEntries.put("h.0.ln_2.weight", new int[] {hiddenSize});
        testEntries.put("h.0.ln_2.bias", new int[] {hiddenSize});
        testEntries.put("h.0.mlp.c_fc.weight", new int[] {hiddenSize, feedForwardSize});
        testEntries.put("h.0.mlp.c_fc.bias", new int[] {feedForwardSize});
        testEntries.put("h.0.mlp.c_proj.weight", new int[] {feedForwardSize, hiddenSize});
        testEntries.put("h.0.mlp.c_proj.bias", new int[] {hiddenSize});

        return testEntries;
    }

    private static LinkedHashMap<String, int[]> getTestEntriesGPT2()
    {
        int tokenCount = 10;
        int hiddenSize = 12;
        int contextSize = 10;
        int feedForwardSize = hiddenSize * 4;

        LinkedHashMap<String, int[]> testEntries = new LinkedHashMap<>();

        testEntries.put("wte.weight", new int[] {tokenCount, hiddenSize});
        testEntries.put("wpe.weight", new int[] {contextSize, hiddenSize});
        testEntries.put("ln_f.weight", new int[] {hiddenSize});
        testEntries.put("ln_f.bias", new int[] {hiddenSize});

        testEntries.put("h.0.ln_1.weight", new int[] {hiddenSize});
        testEntries.put("h.0.ln_1.bias", new int[] {hiddenSize});
        testEntries.put("h.0.attn.c_attn.weight", new int[] {hiddenSize, hiddenSize * 3});
        testEntries.put("h.0.attn.c_attn.bias", new int[] {hiddenSize * 3});
        testEntries.put("h.0.attn.c_proj.weight", new int[] {hiddenSize, hiddenSize});
        testEntries.put("h.0.attn.c_proj.bias", new int[] {hiddenSize});
        testEntries.put("h.0.ln_2.weight", new int[] {hiddenSize});
        testEntries.put("h.0.ln_2.bias", new int[] {hiddenSize});
        testEntries.put("h.0.mlp.c_fc.weight", new int[] {hiddenSize, feedForwardSize});
        testEntries.put("h.0.mlp.c_fc.bias", new int[] {feedForwardSize});
        testEntries.put("h.0.mlp.c_proj.weight", new int[] {feedForwardSize, hiddenSize});
        testEntries.put("h.0.mlp.c_proj.bias", new int[] {hiddenSize});

        return testEntries;
    }

    private static LinkedHashMap<String, int[]> getTestEntriesGPTNEO()
    {
        int tokenCount = 10;
        int hiddenSize = 12;
        int contextSize = 10;
        int feedForwardSize = hiddenSize * 4;

        LinkedHashMap<String, int[]> testEntries = new LinkedHashMap<>();

        testEntries.put("transformer.wte.weight", new int[] {tokenCount, hiddenSize});
        testEntries.put("transformer.wpe.weight", new int[] {contextSize, hiddenSize});
        testEntries.put("transformer.ln_f.weight", new int[] {hiddenSize});
        testEntries.put("transformer.ln_f.bias", new int[] {hiddenSize});

        testEntries.put("transformer.h.0.ln_1.weight", new int[] {hiddenSize});
        testEntries.put("transformer.h.0.ln_1.bias", new int[] {hiddenSize});
        testEntries.put("transformer.h.0.attn.attention.q_proj.weight", new int[] {hiddenSize, hiddenSize});
        testEntries.put("transformer.h.0.attn.attention.k_proj.weight", new int[] {hiddenSize, hiddenSize});
        testEntries.put("transformer.h.0.attn.attention.v_proj.weight", new int[] {hiddenSize, hiddenSize});
        testEntries.put("transformer.h.0.attn.attention.out_proj.weight", new int[] {hiddenSize, hiddenSize});
        testEntries.put("transformer.h.0.attn.attention.out_proj.bias", new int[] {hiddenSize});

        testEntries.put("transformer.h.0.ln_2.weight", new int[] {hiddenSize});
        testEntries.put("transformer.h.0.ln_2.bias", new int[] {hiddenSize});
        testEntries.put("transformer.h.0.mlp.c_fc.weight", new int[] {hiddenSize, feedForwardSize});
        testEntries.put("transformer.h.0.mlp.c_fc.bias", new int[] {feedForwardSize});
        testEntries.put("transformer.h.0.mlp.c_proj.weight", new int[] {feedForwardSize, hiddenSize});
        testEntries.put("transformer.h.0.mlp.c_proj.bias", new int[] {hiddenSize});

        return testEntries;
    }

    private static void generate(String path, String inputSafetensors, String outputSafetensors,
                                 LinkedHashMap<String, int[]> testEntries) throws IOException
    {
        // Create JSON header from the specified test entries
        String header = getHeader(testEntries);
        long headerSize = header.length();

        // Read the source safetensors file
        ParameterReader reader = new ParameterReader(path);
        reader.readSafetensorsModel(path + "/" + inputSafetensors);

        // Write the test file
        File output = new File(path + "/" + outputSafetensors);
        try (DataOutputStream out = new DataOutputStream(new FileOutputStream(output)))
        {
            // Write the length of the header (long)
            out.writeLong(toLittleEndian(headerSize));

            // Write the header (string)
            out.write(header.getBytes(StandardCharsets.UTF_8));

            // Write the tensor values (float)
            for (Map.Entry<String, int[]> entry : testEntries.entrySet())
            {
                int[] shape = entry.getValue();
                if (shape.length == 1)
                {
                    // Write a vector
                    float[] values = reader.readVector(entry.getKey(), shape[0]);
                    for (float value : values)
                    {
                        out.writeFloat(toLittleEndian(value));
                    }
                }
                else if (shape.length == 2)
                {
                    // Write a matrix
                    float[][] values = reader.readMatrix(entry.getKey(), shape[0], shape[1]);
                    for (float[] row : values)
                    {
                        for (float value : row)
                        {
                            out.writeFloat(toLittleEndian(value));
                        }
                    }
                }
            }
        }
    }

    private static String getHeader(Map<String, int[]> descriptions)
    {
        long startOffset = 0;

        Map<String, String> metadata = new HashMap<>();
        metadata.put("format", "pt");

        SafetensorsModel safetensorsModel = new SafetensorsModel(metadata);

        for (Map.Entry<String, int[]> description : descriptions.entrySet())
        {
            int[] dims = description.getValue();

            int size = dims[0];
            for (int i = 1; i < dims.length; i++)
            {
                size *= dims[i];
            }

            List<Integer> shape = Arrays.stream(dims).boxed().collect(Collectors.toList());

            long endOffset = startOffset + size * 4L;

            TensorModel tensorModel = new TensorModel("F32", shape, startOffset, endOffset);
            safetensorsModel.addTensor(description.getKey(), tensorModel);

            startOffset = endOffset;
        }

        try
        {
            return new ObjectMapper().writeValueAsString(safetensorsModel);
        }
        catch (JsonProcessingException e)
        {
            return "";
        }
    }

    public static long toLittleEndian(long value)
    {
        ByteBuffer buffer = ByteBuffer.allocate(8);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putLong(value);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        return buffer.getLong(0);
    }

    public static float toLittleEndian(float value)
    {
        ByteBuffer buffer = ByteBuffer.allocate(4);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putFloat(value);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        return buffer.getFloat(0);
    }
}
