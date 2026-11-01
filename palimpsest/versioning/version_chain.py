import uuid
import json
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel
from palimpsest.storage.models import Version, MemoryItem, DecisionLog
from palimpsest.storage.db import Database

class ConflictInfo(BaseModel):
    item_id_a: Optional[str] = None
    item_id_b: Optional[str] = None
    item_content_a: str
    item_content_b: str
    resolution: str
    flagged: bool = True

class MergeResult(BaseModel):
    merged_version: Version
    merged_items: List[MemoryItem]
    conflicts: List[ConflictInfo]
    summary_message: str

class VersionChain:
    def __init__(self, db: Database):
        self.db = db

    def create_snapshot(
        self,
        session_id: str,
        topic_label: str,
        parent_id: Optional[str] = None,
        branch: str = "main"
    ) -> Version:
        """Creates a new version snapshot and associates active memory items with it."""
        version = Version(
            parent_id=parent_id,
            topic_label=topic_label,
            branch=branch
        )
        self.db.insert_version(version)

        # Get active items for session
        items = self.db.get_memory_items(session_id=session_id, active_only=True, include_discarded=False)
        for item in items:
            # Clone as immutable snapshot item
            cloned = item.model_copy(update={
                "id": str(uuid.uuid4()),
                "session_id": f"snapshot_{version.id}",
                "version_id": version.id
            })
            self.db.insert_memory_item(cloned)

        self.db.log_decision(DecisionLog(
            session_id=session_id,
            action="create_snapshot",
            metadata_json=json.dumps({"version_id": version.id, "branch": branch, "item_count": len(items)}),
            reason=f"Snapshot created: '{topic_label}' on branch '{branch}'."
        ))
        return version

    def trace_back(self, version_id: str) -> List[Version]:
        """
        Traverses from version_id back through parents to the root ancestor.
        Returns ordered list [root, ..., version].
        """
        chain: List[Version] = []
        curr_id = version_id
        visited = set()

        while curr_id and curr_id not in visited:
            visited.add(curr_id)
            v = self.db.get_version(curr_id)
            if not v:
                break
            chain.append(v)
            curr_id = v.parent_id

        chain.reverse()
        return chain

    def restart(
        self,
        version_id: str,
        session_id: str,
        topic_label: Optional[str] = None
    ) -> Tuple[Version, List[MemoryItem]]:
        """
        Restores memory state to version_id, starting a new milestone from that state.
        """
        target_version = self.db.get_version(version_id)
        if not target_version:
            raise ValueError(f"Version {version_id} not found.")

        # Create a new version snapshot indicating restart
        new_version = Version(
            parent_id=target_version.id,
            topic_label=topic_label or f"Restart from {target_version.topic_label}",
            branch=target_version.branch
        )
        self.db.insert_version(new_version)

        # Fetch items from the target version
        items_at_version = self.db.get_memory_items(version_id=target_version.id, include_discarded=False)
        restored_items: List[MemoryItem] = []

        for it in items_at_version:
            restored = it.model_copy(update={
                "id": str(uuid.uuid4()),
                "session_id": session_id,
                "version_id": None
            })
            self.db.insert_memory_item(restored)
            restored_items.append(restored)

        self.db.log_decision(DecisionLog(
            session_id=session_id,
            action="restart",
            metadata_json=json.dumps({"source_version_id": version_id, "new_version_id": new_version.id}),
            reason=f"Memory state restarted from snapshot: '{target_version.topic_label}'."
        ))
        return new_version, restored_items

    def branch(
        self,
        version_id: str,
        new_branch_name: str,
        topic_label: Optional[str] = None
    ) -> Version:
        """
        Creates a new branch originating at version_id.
        """
        parent_v = self.db.get_version(version_id)
        if not parent_v:
            raise ValueError(f"Base version {version_id} not found.")

        branch_version = Version(
            parent_id=parent_v.id,
            topic_label=topic_label or f"Branch: {new_branch_name} from {parent_v.topic_label}",
            branch=new_branch_name
        )
        self.db.insert_version(branch_version)

        # Copy items across
        items = self.db.get_memory_items(version_id=parent_v.id, include_discarded=False)
        for it in items:
            branched_item = it.model_copy(update={
                "id": str(uuid.uuid4()),
                "version_id": branch_version.id
            })
            self.db.insert_memory_item(branched_item)

        self.db.log_decision(DecisionLog(
            session_id=items[0].session_id if items else "system",
            action="branch",
            metadata_json=json.dumps({"base_version_id": version_id, "branch": new_branch_name}),
            reason=f"Created new branch '{new_branch_name}' branching from '{parent_v.topic_label}'."
        ))
        return branch_version

    def merge(
        self,
        branch_a: str,
        branch_b: str,
        target_branch: Optional[str] = None,
        topic_label: Optional[str] = None,
        session_id: str = "session_merge"
    ) -> MergeResult:
        """
        Merges branch_a and branch_b into target_branch (defaults to branch_a).
        Conflict Resolution Rules:
        1. Prefer 'verbatim' over 'summary'.
        2. Keep both if independent.
        3. Flag conflicts if contradictory or same source item has conflicting labels.
        """
        target_branch = target_branch or branch_a

        # Get latest versions on both branches
        all_versions = self.db.get_all_versions()
        versions_a = [v for v in all_versions if v.branch == branch_a]
        versions_b = [v for v in all_versions if v.branch == branch_b]

        if not versions_a or not versions_b:
            raise ValueError(f"Both branches must have at least one version to merge (found {branch_a}:{len(versions_a)}, {branch_b}:{len(versions_b)}).")

        tip_a = versions_a[-1]
        tip_b = versions_b[-1]

        items_a = self.db.get_memory_items(version_id=tip_a.id, include_discarded=False)
        items_b = self.db.get_memory_items(version_id=tip_b.id, include_discarded=False)

        merged_version = Version(
            parent_id=tip_a.id,
            topic_label=topic_label or f"Merge branch '{branch_b}' into '{target_branch}'",
            branch=target_branch
        )
        self.db.insert_version(merged_version)

        conflicts: List[ConflictInfo] = []
        merged_items: List[MemoryItem] = []

        # Index items by original_content or content for overlap detection
        def normalize_key(item: MemoryItem) -> str:
            raw = item.original_content or item.content
            return " ".join(raw.lower().strip().split()[:8])

        dict_a: Dict[str, MemoryItem] = {normalize_key(it): it for it in items_a}
        dict_b: Dict[str, MemoryItem] = {normalize_key(it): it for it in items_b}

        all_keys = set(dict_a.keys()).union(set(dict_b.keys()))

        for k in all_keys:
            in_a = dict_a.get(k)
            in_b = dict_b.get(k)

            if in_a and not in_b:
                # Unique to A
                new_item = in_a.model_copy(update={"id": str(uuid.uuid4()), "version_id": merged_version.id, "session_id": session_id})
                self.db.insert_memory_item(new_item)
                merged_items.append(new_item)
            elif in_b and not in_a:
                # Unique to B
                new_item = in_b.model_copy(update={"id": str(uuid.uuid4()), "version_id": merged_version.id, "session_id": session_id})
                self.db.insert_memory_item(new_item)
                merged_items.append(new_item)
            else:
                # Overlap / Potential Conflict!
                assert in_a is not None and in_b is not None
                
                # Rule: Prefer verbatim over summary
                if in_a.label == "verbatim" and in_b.label == "summary":
                    chosen = in_a
                    conflicts.append(ConflictInfo(
                        item_id_a=in_a.id,
                        item_id_b=in_b.id,
                        item_content_a=in_a.content,
                        item_content_b=in_b.content,
                        resolution="Preferred verbatim from branch A over summary from branch B.",
                        flagged=False
                    ))
                elif in_b.label == "verbatim" and in_a.label == "summary":
                    chosen = in_b
                    conflicts.append(ConflictInfo(
                        item_id_a=in_a.id,
                        item_id_b=in_b.id,
                        item_content_a=in_a.content,
                        item_content_b=in_b.content,
                        resolution="Preferred verbatim from branch B over summary from branch A.",
                        flagged=False
                    ))
                elif in_a.content.strip() == in_b.content.strip():
                    chosen = in_a
                else:
                    # Contradictory / Divergent content on both sides -> flag conflict!
                    # Prefer higher importance or verbatim
                    chosen = in_a if in_a.importance >= in_b.importance else in_b
                    conflicts.append(ConflictInfo(
                        item_id_a=in_a.id,
                        item_id_b=in_b.id,
                        item_content_a=in_a.content,
                        item_content_b=in_b.content,
                        resolution=f"Conflict flagged: Divergent content between branches. Kept item with higher importance ({chosen.importance}).",
                        flagged=True
                    ))

                new_item = chosen.model_copy(update={"id": str(uuid.uuid4()), "version_id": merged_version.id, "session_id": session_id})
                self.db.insert_memory_item(new_item)
                merged_items.append(new_item)

        self.db.log_decision(DecisionLog(
            session_id=session_id,
            action="merge",
            metadata_json=json.dumps({
                "branch_a": branch_a,
                "branch_b": branch_b,
                "merged_version_id": merged_version.id,
                "total_items": len(merged_items),
                "conflicts_count": len(conflicts),
                "flagged_conflicts": sum(1 for c in conflicts if c.flagged)
            }),
            reason=f"Merged branch '{branch_b}' into '{branch_a}' ({len(conflicts)} conflict resolutions, {sum(1 for c in conflicts if c.flagged)} flagged)."
        ))

        return MergeResult(
            merged_version=merged_version,
            merged_items=merged_items,
            conflicts=conflicts,
            summary_message=f"Merged successfully: {len(merged_items)} items total, {len(conflicts)} resolved conflicts."
        )
