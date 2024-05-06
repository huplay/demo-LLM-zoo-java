package huplay.demo.tokenizer;

public class Token
{
    private final int id;
    private String text;

    public Token(int id, String text)
    {
        this.id = id;
        this.text = text;
    }

    // Getters, setters
    public int getId() {return id;}
    public String getText() {return text;}
    public void setText(String text) {this.text = text;}
}
