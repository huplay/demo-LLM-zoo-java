package huplay.demo.config;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.demo.IdentifiedException;
import huplay.demo.util.FloatType;
import huplay.demo.util.Vector;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static huplay.demo.AppLoader.UTIL;
import static huplay.demo.config.SafetensorsModel.TensorModel;

/**
 * Reader of the trained parameters
 */
public class ParameterReader
{
    private final Map<String, ParameterDescriptor> parameterDescriptors = new HashMap<>();

    public ParameterReader(String modelPath)
    {
        // Read the header(s) of the safetensors parameter file(s)
        File modelFolder = new File(modelPath);

        if (!modelFolder.isDirectory())
        {
            throw new IdentifiedException("Model folder not found: " + modelPath);
        }

        for (File file : modelFolder.listFiles())
        {
            if (file.isFile() && file.getName().endsWith("safetensors"))
            {
                readSafetensorsModel(modelPath + "/" + file.getName());
            }
        }
    }

    public void readSafetensorsModel(String fileName)
    {
        long headerSize = readHeaderSize(fileName);
        String header = readHeader(fileName, headerSize);

        Map<String, TensorModel> tensors = new HashMap<>();

        try
        {
            TypeReference<Map<String, TensorModel>> typeRef = new TypeReference<>(){};
            tensors.putAll(new ObjectMapper().readValue(header, typeRef));
        }
        catch (JsonProcessingException e)
        {
            throw new IdentifiedException("Parameter file read error. (" + fileName + ")", e);
        }

        for (Map.Entry<String, TensorModel> entry : tensors.entrySet())
        {
            String id = entry.getKey();

            if (id.equals("__metadata__")) continue;

            TensorModel tensor = entry.getValue();

            DataType dataType = DataType.valueOf(tensor.getDataType());
            List<Integer> shape = tensor.getShape();
            List<Long> offsets = tensor.getDataOffsets();

            if (offsets == null || offsets.size() != 2)
            {
                throw new IdentifiedException("Parameter file read error. (" + id + ")");
            }

            ParameterDescriptor descriptor = new ParameterDescriptor(fileName, id, headerSize + 8, "pt",
                    dataType, shape, offsets.get(0), offsets.get(1));

            parameterDescriptors.put(id, descriptor);
        }
    }

    private long readHeaderSize(String fileName)
    {
        long[] array = new long[1];

        try (FileInputStream stream = new FileInputStream(fileName))
        {
            FileChannel channel = stream.getChannel();
            ByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, 8);
            buffer.order(ByteOrder.LITTLE_ENDIAN);
            LongBuffer longBuffer = buffer.asLongBuffer();

            longBuffer.get(array, 0, 1);
        }
        catch (Exception e)
        {
            throw new IdentifiedException("Parameter file read error. (" + fileName + ")", e);
        }

        return array[0];
    }

    private String readHeader(String fileName, long headerSize)
    {
        byte[] array = new byte[(int)headerSize];

        try (FileInputStream stream = new FileInputStream(fileName))
        {
            FileChannel channel = stream.getChannel();
            ByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 8, headerSize);
            buffer.order(ByteOrder.BIG_ENDIAN);
            ByteBuffer byteBuffer = buffer.asReadOnlyBuffer();

            byteBuffer.get(array, 0, (int)headerSize);
        }
        catch (Exception e)
        {
            throw new IdentifiedException("Parameter file read error. (" + fileName + ")", e);
        }

        return new String(array, StandardCharsets.UTF_8);
    }

    public Vector readVector(String file, int size)
    {
        return read(file, size, false);
    }

    public Vector readVectorOptional(String file, int size)
    {
        return read(file, size, true);
    }

    public Vector[] readMatrix(String file, int rows, int cols)
    {
        Vector vector = read(file, rows * cols, false);
        return vector == null ? null : UTIL.splitVector(vector, rows);
    }

    public Vector[] readMatrixOptional(String file, int rows, int cols)
    {
        Vector vector = read(file, rows * cols, true);
        return vector == null ? null : UTIL.splitVector(vector, rows);
    }

    private void checkSize(ParameterDescriptor descriptor, long expectedSize)
    {
        long parameterSize = descriptor.getSizeInBytes() * 8 / descriptor.getDataType().getBits();
        if (parameterSize != expectedSize)
        {
            System.out.println("\nWARNING: The file has different size (" + parameterSize + ") " +
                    "to the expected (" + expectedSize + "). Id: " + descriptor.getId());
        }
    }

    private Vector read(String id, int size, boolean isOptional)
    {
        ParameterDescriptor descriptor = parameterDescriptors.get(id);

        if (descriptor == null)
        {
            if (isOptional)
            {
                return null;
            }
            else
            {
                throw new IdentifiedException("Descriptor not found for key: " + id);
            }
        }

        checkSize(descriptor, size);

        long offset = descriptor.getDataOffset() + descriptor.getStartOffset();
        File file = new File(descriptor.getFileName());

        try (FileInputStream stream = new FileInputStream(file))
        {
            switch (descriptor.getDataType())
            {
                case F16: return readFloat16(stream, size, offset);
                case BF16: return readBrainFloat16(stream, size, offset);
                case F32: return readFloat32(stream, size, offset);
                default:
                    throw new IdentifiedException("Not supported data type: " + descriptor.getDataType() + ", key: " + id);
            }
        }
        catch (IOException e)
        {
            throw new RuntimeException("Parameter file read error in " + descriptor.getFileName() + ", key: " + id);
        }
    }

    private Vector readFloat32(FileInputStream stream, int size, long offset) throws IOException
    {
        float[] array = new float[size];

        ByteBuffer buffer = stream.getChannel().map(FileChannel.MapMode.READ_ONLY, offset, (long) size * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.asFloatBuffer().get(array, 0, size);

        return new Vector(FloatType.FLOAT32, array);
    }

    private Vector readFloat16(FileInputStream stream, int size, long offset) throws IOException
    {
        short[] array = new short[size];

        ByteBuffer buffer = stream.getChannel().map(FileChannel.MapMode.READ_ONLY, offset, (long) size * 2);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.asShortBuffer().get(array, 0, size);

        float[] ret = new float[size]; // TODO

        for (int i = 0; i < size; i++)
        {
            ret[i] = toFloat32(array[i]);
        }

        return new Vector(FloatType.FLOAT16, array);
    }

    private Vector readBrainFloat16(FileInputStream stream, int size, long offset) throws IOException
    {
        short[] array = new short[size];

        ByteBuffer buffer = stream.getChannel().map(FileChannel.MapMode.READ_ONLY, offset, (long) size * 2);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.asShortBuffer().get(array, 0, size);

        float[] ret = new float[size];

        for (int i = 0; i < size; i++)
        {
            ret[i] = toFloat32(array[i]);
        }

        return new Vector(FloatType.BFLOAT16, array);
    }

    private float toFloat32(short value)
    {
        int signFlag = value & 0b1000_0000_0000_0000; // Extract sign (1st bit)
        int exponent = value & 0b0111_1100_0000_0000; // Extract exponent (5 bits after exponent
        int mantissa = value & 0b0000_0011_1111_1111; // Extract mantissa (last 10 bits)

        if (exponent == 0b0111_1100_0000_0000)
        {
            // Infinity or NaN
            if (mantissa == 0)
            {
                if (signFlag == 0) return Float.POSITIVE_INFINITY;
                else return Float.NEGATIVE_INFINITY;
            }
            else return Float.NaN;
        }
        else if (exponent == 0)
        {
            // Zero or subnormal value
            if (mantissa != 0)
            {
                exponent = 0x1c400;
                do
                {
                    mantissa <<= 1;
                    exponent -= 0b0000_0100_0000_0000;
                }
                while ((mantissa & 0b0000_0100_0000_0000) == 0);

                mantissa &= 0b0000_0011_1111_1111;
            }

            return Float.intBitsToFloat(signFlag << 16 | (exponent | mantissa) << 13);
        }
        else
        {
            // Normal value
            exponent += 0x1c000;
            if (mantissa == 0 && exponent > 0x1c400)
            {
                return Float.intBitsToFloat(signFlag << 16 | exponent << 13 | 0b0000_0011_1111_1111);
            }

            return Float.intBitsToFloat(signFlag << 16 | (exponent | mantissa) << 13);
        }
    }
}
