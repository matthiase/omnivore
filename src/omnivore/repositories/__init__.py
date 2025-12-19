"""
Repositories: Data Access Layer

This package contains repository classes, each responsible for encapsulating all data access logic for a specific entity or aggregate (e.g., Instrument, Prediction, Model). The repository pattern abstracts the details of how data is retrieved, stored, or queried from the underlying database or other data sources.

Repositories should:
- Provide methods for CRUD operations and complex queries related to their entity.
- Contain no business logic—only data access and mapping between database rows and Python objects/dicts.

Distinction from Services:
- **Repositories** answer: "How do I get or store this data?"
- **Services** (in the `services` package) answer: "What should happen in the business when X occurs?" They orchestrate workflows, enforce business rules, and may use multiple repositories to fulfill a use case.

Example:
    - `InstrumentRepository`: Handles all SQL for the `instruments` table.
    - `InstrumentService`: Coordinates instrument creation, validation, and downstream effects, using `InstrumentRepository` for data access.

This separation improves maintainability, testability, and clarity as your project grows.
"""