package huplay.demo.config;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static huplay.demo.App.UTIL;

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

    private void readDescriptor(String fileName) {
        fileName = config.getModelPath() + fileName;

        long headerSize = readHeaderSize(fileName);
        String header = readHeader(fileName, headerSize);

        int index = 0;

        String rawMetadata = null;
        Map<String, String> rawEntries = new HashMap<>();

        while (true) {
            int start = header.indexOf('"', index);

            if (start < 0) {
                break;
            }

            int end = header.indexOf('"', start + 1);

            String key = header.substring(start + 1, end);

            start = header.indexOf('{', end + 1);
            index = header.indexOf('}', start + 1);

            String value = header.substring(start + 1, index);

            if (key.equals(METADATA_KEY)) {
                rawMetadata = value;
            } else {
                rawEntries.put(key, value);
            }

            if (index == headerSize - 1) {
                break;
            }
        }

        String format = readFormat(rawMetadata);

        for (Map.Entry<String, String> entry : rawEntries.entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue();

            DataType dataType = readDataType(value);
            List<Integer> shape = readShape(value);
            String[] offsets = readOffsets(value).split(",");
            long start = Long.parseLong(offsets[0]);
            long end = Long.parseLong(offsets[1]);

            ParameterDescriptor descriptor =
                    new ParameterDescriptor(fileName, headerSize + 8, format, dataType, shape, start, end);

            parameterDescriptors.put(key, descriptor);
        }
    }

    private String readFormat(String value)
    {
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

    private List<Integer> readShape(String value)
    {
        // TODO
        return null;
    }

    private String readOffsets(String value)
    {
        int start = value.indexOf("\"" + OFFSETS_KEY + "\"");

        if (start < 0) return null;

        start = value.indexOf('[', start + OFFSETS_KEY.length() + 2);
        int end = value.indexOf(']', start + 1);

        return value.substring(start + 1, end);
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
        return read(file, size);
    }

    public float[][] readMatrix(String file, int rows, int cols)
    {
        float[] vector = read(file, rows * cols);
        return vector == null ? null : UTIL.splitVector(vector, rows);
    }

    private float[] read(String key, int size)
    {
        ParameterDescriptor descriptor = parameterDescriptors.get(key);

        if (descriptor == null)
        {
            throw new RuntimeException("Descriptor not found for key: " + key);
        }

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
            //ret[i] = toFullPrecision(array[i], array[i+1]);
        }

        return ret;
    }

/*
    private float toFullPrecision(byte first, byte second)
    {
        ByteBuffer byteBuffer = ByteBuffer.allocate(4);

        byteBuffer.put((byte)0);
        byteBuffer.put((byte)0);
        byteBuffer.put(first);
        byteBuffer.put(second);

        byteBuffer.position(0);

        return byteBuffer.getFloat();
    }

    private float toFloat32(short value)
    {
        int mantisa = value & 0x03ff;
        int exponent = value & 0x7c00;

        if (exponent == 0x7c00)
        {
            exponent = 0x3fc00;
        }
        else if (exponent != 0)
        {
            exponent += 0x1c000;
            if (mantisa == 0 && exponent > 0x1c400)
            {
                return Float.intBitsToFloat((value & 0x8000) << 16 | exponent << 13 | 0x3ff);
            }
        }
        else if (mantisa != 0)
        {
            exponent = 0x1c400;
            do
            {
                mantisa <<= 1;
                exponent -= 0x400;
            }
            while ((mantisa & 0x400) == 0);

            mantisa &= 0x3ff;
        }

        return Float.intBitsToFloat((value & 0x8000) << 16 | (exponent | mantisa) << 13);
    }
*/
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

    public float[][] splitVectorTransposed(float[] numbers, int rows, int cols)
    {
        float[][] matrix = new float[rows][cols];

        int row = 0;
        int col = 0;
        for (int i = 0; i < numbers.length; i++)
        {
            matrix[row][col] = numbers[i];

            row++;

            if (row == rows)
            {
                row = 0;
                col++;
            }
        }

        return matrix;
    }
/*
    private File findFile(String name, int size)
    {
        String fileName = config.getModelPath() + name;
        File file = new File(fileName);

        if ( ! file.exists())
        {
            // Handling files split into parts
            List<File> partFiles = new ArrayList<>();

            int i = 1;
            while (true)
            {
                File partFile = new File(fileName + ".part" + i);

                if (partFile.exists()) partFiles.add(partFile);
                else break;

                i++;
            }

            if (partFiles.isEmpty())
            {
                throw new RuntimeException("Parameter file not found: " + fileName);
            }
            else
            {
                file = mergeAndSaveParts(partFiles, fileName);
            }
        }

        if (file.length() != size)
        {
            throw new RuntimeException("Incorrect file size (" + file.length() + "). Expected: " + size);
        }

        return file;
    }

    private File mergeAndSaveParts(List<File> partFiles, String fileName)
    {
        File file = new File(fileName);

        try
        {
            DataOutputStream output = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(file.toPath())));

            for (File partFile : partFiles)
            {
                byte[] array = new byte[(int) partFile.length()];

                try (FileInputStream stream = new FileInputStream(partFile))
                {
                    FileChannel inChannel = stream.getChannel();
                    ByteBuffer buffer = inChannel.map(FileChannel.MapMode.READ_ONLY, 0, inChannel.size());

                    buffer.get(array);
                }
                catch (Exception e)
                {
                    throw new RuntimeException("Parameter file read error. (" + partFile.getName() + ")");
                }

                for (byte value : array)
                {
                    output.writeByte(value);
                }
            }

            output.close();
        }
        catch (IOException e)
        {
            throw new RuntimeException("Can't create concatenated file (" + fileName + ") Exception: " + e.getMessage());
        }

        return file;
    }
*/
}
