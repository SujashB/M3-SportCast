import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
import fitz
import spacy
from tqdm import tqdm

load_dotenv()

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class FencingKnowledgeExtractor:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.nlp = spacy.load("en_core_web_sm")
        
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def create_knowledge_nodes(self, text, book_title):
        """Create knowledge nodes from text"""
        doc = self.nlp(text)
        
        # Extract fencing-specific entities and relationships
        with self.driver.session() as session:
            # Create book node
            session.run(
                "CREATE (b:Book {title: $title})",
                title=book_title
            )
            
            # Process sentences and create relationships
            for sent in doc.sents:
                # Extract fencing techniques, movements, and concepts
                entities = []
                for ent in sent.ents:
                    if self.is_fencing_related(ent.text):
                        entities.append(ent)
                
                # Create nodes and relationships
                for i, ent in enumerate(entities):
                    # Create or merge entity node
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        ON CREATE SET e.type = $type
                        WITH e
                        MATCH (b:Book {title: $book})
                        MERGE (e)-[:MENTIONED_IN]->(b)
                        """,
                        name=ent.text,
                        type=ent.label_,
                        book=book_title
                    )
                    
                    # Create relationships between consecutive entities
                    if i > 0:
                        session.run(
                            """
                            MATCH (e1:Entity {name: $name1})
                            MATCH (e2:Entity {name: $name2})
                            MERGE (e1)-[:RELATED_TO]->(e2)
                            """,
                            name1=entities[i-1].text,
                            name2=ent.text
                        )
    
    def is_fencing_related(self, text):
        """Check if text is related to fencing"""
        fencing_terms = {
            'attack', 'defense', 'parry', 'riposte', 'lunge', 'advance',
            'retreat', 'blade', 'point', 'guard', 'stance', 'footwork',
            'distance', 'tempo', 'bout', 'target', 'line', 'engagement',
            'fencing', 'fencer', 'sword', 'sabre', 'epee', 'foil',
            'technique', 'movement', 'position', 'action', 'motion'
        }
        return any(term in text.lower() for term in fencing_terms)

def process_fencing_books(books_dir, knowledge_extractor):
    """Process all fencing books and create knowledge graph"""
    print("Processing fencing books...")
    for book_file in os.listdir(books_dir):
        if book_file.endswith('.pdf'):
            print(f"Processing {book_file}...")
            book_path = os.path.join(books_dir, book_file)
            text = knowledge_extractor.extract_text_from_pdf(book_path)
            knowledge_extractor.create_knowledge_nodes(text, book_file)
    print("Knowledge graph creation complete!")

if __name__ == "__main__":
    # Initialize knowledge extractor
    knowledge_extractor = FencingKnowledgeExtractor()
    
    # Clear existing database
    print("Clearing existing database...")
    knowledge_extractor.clear_database()
    
    # Process fencing books
    process_fencing_books("fencing_books", knowledge_extractor)
    
    # Close connection
    knowledge_extractor.close()
    print("Database population complete!") 