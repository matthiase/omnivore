require "nokogiri"
require "omnivore/http_client"

module Omnivore

  # A class encapsulating an HTML document.
  class Document
    attr_reader :model

    # The HTML tags signaling the start of a block or paragraph.
    BLOCK_TAGS = %w[div p frame]

    # A Struct descibing a paragraph, including it's :path in the document, :text,
    # and various metrics, such as :text_density.
    Paragraph = Struct.new("Paragraph", :path, :text, :text_density)


    # Creates a Omnivore::Document object from a url.
    # @param [String] url the document's url
    # @return [Document] A new Document object.
    def self.from_url(url)
      Document.new(HttpClient.get(url))
    end


    # Creates a Omnivore::Document object from a string containing HTML.
    # @param [String] html the HTML content
    # @return [Document] A new Document object.
    def self.from_html(html)
      Document.new(html)
    end


    def initialize(html)
      @model = Nokogiri::HTML.parse(html) { |config|
        config.options = Nokogiri::XML::ParseOptions::NOBLANKS
      }
    end


    # A HTML representation of the document.
    # @return [String] A HTML representation of the document.
    def to_html
      self.model.to_html
    end


    # Extracts the document title.
    # @return [String] The document title.
    def title
      @title ||= self.model.xpath("/html/head/title").text.gsub(/\s+/, " ").strip
    end


    # Extracts document metadata.
    # @return [Hash] The metadata tags found in the document.
    def metadata
      @metadata ||= self.model.xpath("//meta").inject({ }) { |memo, el|
        memo[el.attr("name")] = el.attr("content") || "" if el.attr("name")  
        memo
      }
    end


    # Returns the actual content of the document, without navigation, advertising, etc.
    # @return [String] The document's main content. 
    def to_text
      self.to_paragraphs.inject([ ]) { |buffer, p| 
        buffer << p.text if p.text_density >= 0.25
        buffer
      }.join("\n")
    end


    # Splits the document into paragraphs, assuming that each <div> or <p> tag represents
    # a paragraph.
    # @return [Array] An array of Paragraph objects.
    def to_paragraphs
      self.model.xpath("//div|//p").map { |block|
        html = block.to_html.gsub(/\s+/, " ").strip
        text = flatten(block).inject([ ]) { |memo, node|
          memo << node.text.gsub(/\s+/, " ").strip if node.kind_of?(Nokogiri::XML::Text) 
          memo
        }.join(" ")
        Paragraph.new(block.path.to_s, text, text.size / html.size.to_f)
      }
    end


    private

    # A convenience method that recursively iterates over a document node and returns
    # an array of all of it's children, with the exception of other block elements 
    # (e.g div or p nodes).
    # @param [Nokogiri::XML::Node] node the root node
    # @return [Array] The Nokogiri::XML::Node objects contained in the root.
    def flatten(node)
      elements = [ ]
      return elements if node.nil?
      return elements if node.respond_to?('cdata?') and node.cdata?
      return elements if node.respond_to?('comment?') and node.comment?
      if node.children.empty?
        elements << node
      else
        node.children.each { |child|
          unless BLOCK_TAGS.include?(child.name)
            elements += flatten(child)
          end
        }
      end
      elements
    end

  end

end
