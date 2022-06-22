x: str="abc123fdskf"
dg = int("".join(filter(lambda c:c.isdigit() ,x)))
print(dg, type(dg))
