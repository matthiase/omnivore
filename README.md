## Omnivore: a library for decrufting HTML documents

Omnivore is a library for extracting "real" content from HTML documents.  Currently, the approach is limited to
analyzing text density to distinguish relevant sections from navigation, advertising, and other non-relevant elements. As
such, the results are far from perfect but will hopefully improve as more sophisticated features are added.

### DEPENDENCIES
**Runtime**
- Nokogiri

**Development**
- Red Carpet
- RSpec
- Yard

### INSTALL
```
gem install omnivore
```

### EXAMPLE
```ruby
require 'omnivore'
document = Omnivore::Document.from_url('http://www.slashgear.com/sennheiser-hd-700-hands-on-10208572')
puts document.to_text
```

