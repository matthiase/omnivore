# -*- encoding: utf-8 -*-
$:.push File.expand_path("../lib", __FILE__)
require "omnivore/version"

Gem::Specification.new do |s|
  s.name        = "omnivore"
  s.version     = Omnivore::VERSION
  s.platform    = Gem::Platform::RUBY
  s.authors     = ["Matthias Eder"]
  s.email       = ["matthias@izume.com"]
  s.homepage    = "http://github.com/matthiase/omnivore"
  s.summary     = %q{Content Extraction and Analysis Library}
  s.description = %q{A library for extracting content from HTML documents.}

  s.rubyforge_project = "omnivore"

  s.files         = `git ls-files`.split("\n")
  s.test_files    = `git ls-files -- {test,spec,features}/*`.split("\n")
  s.executables   = `git ls-files -- bin/*`.split("\n").map{ |f| File.basename(f) }
  s.require_paths = ["lib"]

  # specify any dependencies here; for example:
  s.add_development_dependency "yard", "~> 0.7.4"
  s.add_development_dependency "redcarpet", "~> 2.0.1"
  s.add_development_dependency "rspec", "~> 2.8.0"
  s.add_runtime_dependency "nokogiri", "~> 1.5.0"
end
