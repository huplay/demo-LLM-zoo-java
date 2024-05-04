package huplay.demo;

import huplay.demo.config.ModelConfig;

import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.net.URLConnection;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

public class Downloader implements Runnable
{
    private static final int BATCH_SIZE = 5120;

    private final String fileName;
    private final String path;
    private final URL url;

    private boolean isInProgress = true;
    private final long size;
    private long pieces;
    private long position = 0;
    private boolean isOk = true;

    public Downloader(ModelConfig modelConfig, String fileName, String path) throws Exception
    {
        this.fileName = fileName;
        this.path = path;

        url = new URL(determineDownloadUrl(modelConfig, fileName));
        size = getSize(url);

        pieces = Math.floorDiv(size, BATCH_SIZE);
        if (size % BATCH_SIZE > 0) pieces++;
    }

    @Override
    public void run()
    {
        try (FileOutputStream outputStream = new FileOutputStream(path + "/" + fileName))
        {
            ReadableByteChannel urlChannel = Channels.newChannel(url.openStream());
            FileChannel fileChannel = outputStream.getChannel();

            long startPos = 0;
            for (position = 0; position < pieces; position++)
            {
                fileChannel.transferFrom(urlChannel, startPos, BATCH_SIZE);
                startPos += BATCH_SIZE;
            }
        }
        catch (IOException e)
        {
            isOk = false;
            throw new RuntimeException("IO error during file download: " + fileName + " error: " + e);
        }

        isInProgress = false;
    }

    private long getSize(URL url) throws IOException
    {
        URLConnection urlConnection = url.openConnection();
        urlConnection.connect();
        return urlConnection.getContentLengthLong();
    }

    private String determineDownloadUrl(ModelConfig modelConfig, String fileName)
    {
        String repoUrl = modelConfig.getRepo();

        if (repoUrl.startsWith("https://huggingface.co/"))
        {
            String branch = modelConfig.getBranch();

            if (branch == null || branch.equals(""))
            {
                branch = "main";
            }

            repoUrl += "/resolve/" + branch + "/" + fileName + "?download=true";
        }

        return repoUrl;
    }

    // Getters
    public boolean isInProgress() {return isInProgress;}
    public boolean isOk() {return isOk;}
    public long getSize() {return size;}
    public long getPieces() {return pieces;}
    public long getPosition() {return position;}
}
