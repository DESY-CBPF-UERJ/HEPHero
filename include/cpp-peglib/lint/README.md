peglint
-------

The lint utility for PEG.

```
usage: grammar_file_path [source_file_path]

  options:
    --ast: show AST tree
    --packrat: enable packrat memoise
    --opt, --opt-all: optimaze all AST nodes except nodes selected with `no_ast_opt` instruction
    --opt-only: optimaze only AST nodes selected with `no_ast_opt` instruction
    --source: source text
    --trace: show trace messages
```

### Build peglint

```
> cd lint
> mkdir build
> cd build
> cmake ..
> make
```

### Lint grammar

```
> cat a.peg
A <- 'hello' ^ 'world'

> peglint a.peg
a.peg:1:16: syntax error
```

```
> cat a.peg
A <- B

> peglint a.peg
a.peg:1:6: 'B' is not defined.
```

```
> cat a.peg
A <- B / C
B <- 'b'
C <- A

> peglint a.peg
a.peg:1:10: 'C' is left recursive.
a.peg:3:6: 'A' is left recursive.
```

### Lint source text

```
> cat a.peg
Additive    <- Multitive '+' Additive / Multitive
Multitive   <- Primary '*' Multitive / Primary
Primary     <- '(' Additive ')' / Number
Number      <- < [0-9]+ >
%whitespace <- [ \t\r\n]*

> peglint --source "1 + a * 3" a.peg
[commendline]:1:3: syntax error
```

### AST

```
> cat a.txt
1 + 2 * 3

> peglint --ast a.peg a.txt
+ Additive
  + Multitive
    + Primary
      - Number (1)
  + Additive
    + Multitive
      + Primary
        - Number (2)
      + Multitive
        + Primary
          - Number (3)
```

### AST optimazation

```
> peglint --ast --opt --source "1 + 2 * 3" a.peg
+ Additive
  - Multitive[Number] (1)
  + Additive[Multitive]
    - Primary[Number] (2)
    - Multitive[Number] (3)
```

### Adjust AST optimazation with `no_ast_opt` instruction

```
> cat a.peg
Additive    <- Multitive '+' Additive / Multitive
Multitive   <- Primary '*' Multitive / Primary
Primary     <- '(' Additive ')' / Number          { no_ast_opt }
Number      <- < [0-9]+ >
%whitespace <- [ \t\r\n]*

> peglint --ast --opt --source "1 + 2 * 3" a.peg
+ Additive/0
  + Multitive/1[Primary]
    - Number (1)
  + Additive/1[Multitive]
    + Primary/1
      - Number (2)
    + Multitive/1[Primary]
      - Number (3)

> peglint --ast --opt-only --source "1 + 2 * 3" a.peg
+ Additive/0
  + Multitive/1
    - Primary/1[Number] (1)
  + Additive/1
    + Multitive/0
      - Primary/1[Number] (2)
      + Multitive/1
        - Primary/1[Number] (3)
```
