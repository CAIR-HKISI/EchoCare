
class AtlasNode:
    registry = {}

    def __init__(self, node_id, name):
        self.id = node_id
        self.name = name
        self.parent = None
        self.children = []
        AtlasNode.registry[self.id] = self

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def get_parents(self):
        parents = []
        node = self.parent
        while node:
            parents.append(node.id)
            node = node.parent
        return parents

    def get_children(self):
        result = []
        def dfs(n):
            for child in n.children:
                result.append(child.id)
                dfs(child)
        dfs(self)
        return result

    def get_direct_children(self):
        """Return direct children (one level down)."""
        return [child.id for child in self.children]

    def get_direct_parent(self):
        """Return direct parent (one level up)."""
        return self.parent.id if self.parent else None

    @classmethod
    def get_direct_parent_by_id(cls, node_id):
        node = cls.get_by_id(node_id)
        return node.get_direct_parent() if node else None

    @classmethod
    def get_direct_children_by_id(cls, node_id):
        node = cls.get_by_id(node_id)
        return node.get_direct_children() if node else []

    @classmethod
    def get_by_id(cls, node_id):
        return cls.registry.get(node_id)

    @classmethod
    def get_parents_by_id(cls, node_id):
        node = cls.get_by_id(node_id)
        return node.get_parents() if node else []

    @classmethod
    def get_children_by_id(cls, node_id):
        node = cls.get_by_id(node_id)
        return node.get_children() if node else []

    @classmethod
    def get_roots(cls):
        """Return all root nodes (no parent)."""
        return [node.id for node in cls.get_all() if node.parent is None]

    @classmethod
    def get_leaves(cls):
        """Return all leaf nodes (no children)."""
        return [node.id for node in cls.get_all() if len(node.children) == 0]

    @classmethod
    def get_middle_level_nodes(cls):
        """Return all middle level nodes (have children and parent)."""
        result = []
        for node in AtlasNode.get_all():
            if node.parent is not None and len(node.children) > 0:
                result.append(node.id)
        return result

    @classmethod
    def get_all(cls):
        return list(cls.registry.values())

# simple_project/
# ├── dataset/
# │   ├── 0-A2C/
# │   │   ├── 0001.png
# │   │   ├── 0002.png
# │   │   ├── ...
# │   │   └── xxxx.png
# │   ├── 1-A4C/
# │   │   ├── 0001.png
# │   │   ├── 0002.png
# │   │   ├── ...
# │   │   └── xxxx.png
# │   ├── 2-Thyroid/
# │   │   ├── 0001.png
# │   │   ├── 0002.png
# │   │   ├── ...
# │   │   └── xxxx.png
# └──src/


# bottom-level: each class in dataset
A2C = AtlasNode(0, 'A2C')
A4C = AtlasNode(1, 'A4C')
thyroid = AtlasNode(2, 'Thyroid')

# middle-level
heart = AtlasNode(3, 'Heart')

# top-level
thorax = AtlasNode(4, 'thorax')
head_neck = AtlasNode(5, "Head&Neck")

# build the relations
heart.add_child(A2C)
heart.add_child(A4C)
thorax.add_child(heart)

head_neck.add_child(thyroid)

