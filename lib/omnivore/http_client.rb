require 'net/http'
require 'uri'

module Omnivore
  # A simple HTTP client with a redirect feature. 
  #
  class HttpClient

    # Sends a `GET` request to the specified url, following the provided number of 
    # maximum redirects.
    #
    # @param [String] url the url to be requested
    # @param [Integer] redirects the number of redirects to follow
    # @return [String] the response body of the request.
    def self.get(url, redirects=3)
      raise ArgumentError, 'HTTP redirect too deep' if redirects == 0
      response = Net::HTTP.get_response(URI.parse(url))
      case response
      when Net::HTTPSuccess then response.body
      when Net::HTTPRedirection then HttpClient.get(response['location'], redirects - 1)
      else
        response.error!
      end
    end

  end
end
