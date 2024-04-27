package huplay.demo.config;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static huplay.demo.AppLoader.UTIL;

/**
 * Reader of the trained parameters
 */
public class ParameterReader
{
    private static final String METADATA_KEY = "__metadata__";
    private static final String FORMAT_KEY = "format";
    private static final String DATA_TYPE_KEY = "dtype";
    private static final String SHAPE_KEY = "shape";
    private static final String OFFSETS_KEY = "data_offsets";

    private final Config config;

    private final Map<String, ParameterDescriptor> parameterDescriptors = new HashMap<>();

    public ParameterReader(Config config)
    {
        this.config = config;

        // Read the header(s) of the safetensors parameter file(s)
        for (String fileName : config.getParameterFiles())
        {
            // TODO: If missing, download it
            /*
            URL url = new URL(config.getParameterUrl() + "/" + fileName);

            ReadableByteChannel urlChannel = Channels.newChannel(url.openStream());

            FileOutputStream outputStream = new FileOutputStream(fileName);
            FileChannel fileChannel = outputStream.getChannel();

            fileChannel.transferFrom(urlChannel, 0, Long.MAX_VALUE);
            */

            readDescriptor(fileName);
        }
    }

    private void readDescriptor(String fileName)
    {
        fileName = config.getModelPath() + fileName;

        long headerSize = readHeaderSize(fileName);
        String header = readHeader(fileName, headerSize);

        int index = 0;

        String rawMetadata = null;
        Map<String, String> rawEntries = new HashMap<>();

        while (true)
        {
            int start = header.indexOf('"', index);

            if (start < 0)
            {
                break;
            }

            int end = header.indexOf('"', start + 1);

            String key = header.substring(start + 1, end);

            start = header.indexOf('{', end + 1);
            index = header.indexOf('}', start + 1);

            String value = header.substring(start + 1, index);

            if (key.equals(METADATA_KEY))
            {
                rawMetadata = value;
            }
            else
            {
                rawEntries.put(key, value);
            }

            if (index == headerSize - 1)
            {
                break;
            }
        }

        String format = readFormat(rawMetadata);

        for (Map.Entry<String, String> entry : rawEntries.entrySet())
        {
            String id = entry.getKey();
            String value = entry.getValue();

            DataType dataType = readDataType(value);
            List<Long> shape = readShape(value);
            long[] offsets = readOffsets(value);

            if (offsets == null || offsets.length != 2)
            {
                throw new RuntimeException("Parameter file read error. (" + id + ")");
            }

            ParameterDescriptor descriptor = new ParameterDescriptor(fileName, id, headerSize + 8, format,
                    dataType, shape, offsets[0], offsets[1]);

            parameterDescriptors.put(id, descriptor);
        }
    }

    private String readFormat(String value)
    {
        if (value == null)
        {
            return null;
        }

        int start = value.indexOf("\"" + FORMAT_KEY + "\"");

        if (start < 0) return null;

        start = value.indexOf('"', start + FORMAT_KEY.length() + 2);
        int end = value.indexOf('"', start + 1);

        return value.substring(start + 1, end);
    }

    private DataType readDataType(String value)
    {
        int start = value.indexOf("\"" + DATA_TYPE_KEY + "\"");

        if (start < 0) return null;

        start = value.indexOf('"', start + DATA_TYPE_KEY.length() + 2);
        int end = value.indexOf('"', start + 1);

        String dtype = value.substring(start + 1, end);

        return DataType.valueOf(dtype);
    }

    private List<Long> readShape(String value)
    {
        int start = value.indexOf("\"" + SHAPE_KEY + "\"");

        if (start < 0) return null;

        start = value.indexOf('[', start + OFFSETS_KEY.length() + 2);
        int end = value.indexOf(']', start + 1);

        String shape = value.substring(start + 1, end);

        String[] parts = shape.split(",");

        List<Long> result = new ArrayList<>(parts.length);

        for (String part : parts)
        {
            result.add(Long.parseLong(part));
        }

        return result;
    }

    private long[] readOffsets(String value)
    {
        int start = value.indexOf("\"" + OFFSETS_KEY + "\"");

        if (start < 0) return null;

        start = value.indexOf('[', start + OFFSETS_KEY.length() + 2);
        int end = value.indexOf(']', start + 1);

        String offsets = value.substring(start + 1, end);

        String[] parts = offsets.split(",");

        long[] result = new long[2];
        result[0] = Long.parseLong(parts[0]);
        result[1] = Long.parseLong(parts[1]);

        return result;
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
            throw new RuntimeException("Parameter file read error. (" + fileName + ")");
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
            throw new RuntimeException("Parameter file read error. (" + fileName + ")");
        }

        return new String(array, StandardCharsets.UTF_8);
    }

    public float[] readVector(String file, int size)
    {
        return read(file, size, false);
    }

    public float[] readVectorOptional(String file, int size)
    {
        return read(file, size, true);
    }

    public float[][] readMatrix(String file, int rows, int cols)
    {
        float[] vector = read(file, rows * cols, false);
        return vector == null ? null : UTIL.splitVector(vector, rows);
    }

    public float[][] readMatrixOptional(String file, int rows, int cols)
    {
        float[] vector = read(file, rows * cols, true);
        return vector == null ? null : UTIL.splitVector(vector, rows);
    }

    private void checkSize(ParameterDescriptor descriptor, long expectedSize)
    {
        long parameterSize = descriptor.getSizeInBytes() * 8 / descriptor.getDataType().getBits();
        if (parameterSize != expectedSize)
        {
            throw new RuntimeException("The file has different size (" + parameterSize + ") " +
                    "to the expected (" + expectedSize + "). Id: " + descriptor.getId());
        }
    }

    private float[] read(String key, int size, boolean isOptional)
    {
        ParameterDescriptor descriptor = parameterDescriptors.get(key);

        if (descriptor == null)
        {
            if (isOptional)
            {
                return null;
            }
            else
            {
                throw new RuntimeException("Descriptor not found for key: " + key);
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
                    throw new RuntimeException("Not supported data type: " + descriptor.getDataType() + ", key: " + key);
            }
        }
        catch (IOException e)
        {
            throw new RuntimeException("Parameter file read error in " + descriptor.getFileName() + ", key: " + key);
        }
    }

    private float[] readFloat32(FileInputStream stream, int size, long offset) throws IOException
    {
        float[] array = new float[size];

        ByteBuffer buffer = stream.getChannel().map(FileChannel.MapMode.READ_ONLY, offset, (long) size * 4);
        buffer.order(ByteOrder.LITTLE_ENDIAN);
        buffer.asFloatBuffer().get(array, 0, size);

        return array;
    }

    private float[] readFloat16(FileInputStream stream, int size, long offset) throws IOException
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

        return ret;
    }

    private float[] readBrainFloat16(FileInputStream stream, int size, long offset) throws IOException
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

        return ret;
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
