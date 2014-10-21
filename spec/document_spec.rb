require 'omnivore/document'
include Omnivore

describe Document do
  html = File.open("spec/fixtures/thia-breen-interview", "r") { |f| f.readlines }.join("\n")

  it "should fetch the content of the provided url" do
    document = Document.from_url("http://www.google.com")
    document.to_html.should_not be_empty
  end


  it "should contain the document title" do
    document = Document.from_html(html)
    document.title.should_not be_nil
    document.title.should_not be_empty
    document.title.should == "Estee Lauder President Thia Breen Interview - Career Advice from Thia Breen - Marie Claire"

  end


  it "should contain the document metadata" do
    document = Document.from_html(html)
    document.metadata.should_not be_nil
    document.metadata.should_not be_empty
    document.metadata["keywords"].split(",").first.strip.should == "career advice"
  end


  it "should be able to extract the main content and ignore navigation and ads." do
    document = Document.from_html(html)
    text = document.to_text
    text.should_not be_nil
    text.should_not be_empty
  end

end
