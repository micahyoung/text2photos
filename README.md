# text2photos

Apple Photos search script using OpenAI-compatible API and SQLite queries.

## Description

A Python tool that enables intelligent searching through your Apple Photos library using natural language queries powered by OpenAI-compatible APIs.

## Requirements

- Python 3.10+
- macOS
- Apple Photos library

## Installation

```bash
pip install -e .
```

## Dependencies

- pandas
- openai
- osxphotos

## Usage

### Basic usage (creates album "text2photos")
```bash
text2photos -p "photos from NYC with no face masks"
```

### Create custom album
```bash
text2photos -p "highest aesthetic photos of 2024" -a "best_2024_photos"
```

### Export to directory
```bash
text2photos -p "photos of Mom from vacation" -o ./mom_vacation_photos
```

### With custom API settings
```bash
text2photos -p "night photos from NYC" --api-base http://localhost:11434/v1 --model qwen3-coder:30b
```

## License

MIT License