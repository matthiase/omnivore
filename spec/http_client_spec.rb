require 'omnivore/http_client'

describe Omnivore::HttpClient do

  it "should fetch the content of a url" do
    html = Omnivore::HttpClient.get("http://linksmart.com")
    html.should_not be_nil
    html.should_not be_empty
  end

end
