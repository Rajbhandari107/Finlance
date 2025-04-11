from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
import json
from datetime import datetime, timedelta
import uuid
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
app = Flask(__name__)

# Simple data storage (in production, you'd use a real database)
DATA_FILE = 'data.json'

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            # Ensure all required keys exist
            if 'expenses' not in data:
                data['expenses'] = []
            if 'income' not in data:
                data['income'] = []
            if 'clients' not in data:
                data['clients'] = []
            if 'invoices' not in data:
                data['invoices'] = []
            return data
    return {'clients': [], 'invoices': [], 'income': [], 'expenses': []}

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)

# AI Assistant Functions
ai_conversations = {}  # Stores conversation history by session

from transformers import pipeline

def get_ai_response(session_id, user_message):
    """Handle conversational AI interactions with financial expertise"""
    if session_id not in ai_conversations:
        ai_conversations[session_id] = []
    
    data = load_data()
    financial_context = {
        'total_income': sum(item['amount'] for item in data['income']),
        'total_expenses': sum(item['amount'] for item in data['expenses']),
        'estimated_tax': (sum(item['amount'] for item in data['income']) - 
                         sum(item['amount'] for item in data['expenses'])) * 0.3
    }

    # Common financial quick answers
    lower_msg = user_message.lower()
    if 'tax deduct' in lower_msg:
        return "Common deductions: home office, internet, equipment, courses."
    elif 'quarterly tax' in lower_msg:
        return "Pay estimated taxes quarterly (Apr/Jun/Sep/Jan)."
    elif 'invoice late' in lower_msg:
        return "Send reminders, then charge late fees (1-2%/month)."
    elif 'freelance rate' in lower_msg:
        return "Rates vary: $25-200/hr based on experience and skills."
    elif 'emergency fund' in lower_msg:
        return "Save 3-6 months of expenses in separate account."

    # Fall back to AI model
    if not hasattr(app, 'llm_pipe'):
        app.llm_pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device="cpu",
            torch_dtype="auto"
        )
    
    prompt = f"""<|system|>
    You are FinAI, a financial assistant for freelancers. Current status:
    - Income: ${financial_context['total_income']:,.2f}
    - Expenses: ${financial_context['total_expenses']:,.2f}
    - Estimated tax: ${financial_context['estimated_tax']:,.2f}
    Keep responses under 100 words.</s>
    <|user|>
    {user_message}</s>
    <|assistant|>"""
    
    try:
        # Generate response
        response = app.llm_pipe(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )[0]['generated_text']
        
        # Extract just the assistant's response
        assistant_response = response.split("<|assistant|>")[-1].strip()
        
        # Store conversation history
        ai_conversations[session_id].extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ])
        
        return assistant_response
        
    except Exception as e:
        print(f"AI Error: {e}")
        return "I'm having trouble generating a response. Please try again later."

# PDF Generation
def generate_pdf(invoice):
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        story = []
        
        # Invoice Header
        story.append(Paragraph("INVOICE", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Invoice Details
        invoice_data = [
            ["Invoice #:", invoice['id']],
            ["Date:", invoice['date']],
            ["Due Date:", invoice['due_date']],
            ["Status:", invoice['status']],
            ["Client:", invoice['client_name']]
        ]
        
        t = Table(invoice_data, colWidths=[100, 300])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 24))
        
        # Invoice Items
        item_data = [["Description", "Amount"]]
        for item in invoice['items']:
            item_data.append([item['description'], f"${item['amount']:,.2f}"])
        
        item_data.append(["TOTAL", f"${invoice['amount']:,.2f}"])
        
        t = Table(item_data, colWidths=[350, 100])
        t.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('GRID', (0,0), (-1,-2), 1, colors.grey),
            ('GRID', (0,-1), (-1,-1), 1, colors.black),
            ('ALIGN', (1,0), (1,-1), 'RIGHT'),
            ('FONTWEIGHT', (0,-1), (1,-1), 'BOLD'),
        ]))
        story.append(t)
        story.append(Spacer(1, 36))
        
        # Terms and Notes
        story.append(Paragraph("Payment Terms:", styles['Heading3']))
        story.append(Paragraph("Payment due within 30 days. Late payments subject to 1.5% monthly interest.", styles['Normal']))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Notes:", styles['Heading3']))
        story.append(Paragraph("Thank you for your business! Please include invoice number with payment.", styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        return None

# Routes
@app.route('/')
def index():
    data = load_data()
    total_income = sum(item['amount'] for item in data['income'])
    total_expenses = sum(item['amount'] for item in data['expenses'])
    estimated_tax = (total_income - total_expenses) * 0.3  # Tax on profit
    
    return render_template('index.html', 
                         clients=data['clients'],
                         invoices=data['invoices'],
                         income=data['income'],
                         expenses=data['expenses'],
                         total_income=total_income,
                         total_expenses=total_expenses,
                         estimated_tax=estimated_tax,
                         datetime=datetime)

@app.route('/download_invoice/<invoice_id>')
def download_invoice(invoice_id):
    data = load_data()
    invoice = next((i for i in data['invoices'] if i['id'] == invoice_id), None)
    if not invoice:
        return "Invoice not found", 404
    
    pdf = generate_pdf(invoice)
    return send_file(
        pdf,
        as_attachment=True,
        download_name=f"invoice_{invoice_id}.pdf",
        mimetype='application/pdf'
    )

@app.route('/ai/chat', methods=['POST'])
def ai_chat():
    session_id = request.cookies.get('session_id', str(uuid.uuid4()))
    user_message = request.json.get('message', '')
    
    data = load_data()
    financial_context = {
        'total_income': sum(item['amount'] for item in data['income']),
        'total_expenses': sum(item['amount'] for item in data['expenses']),
        'estimated_tax': (sum(item['amount'] for item in data['income']) - 
                          sum(item['amount'] for item in data['expenses'])) * 0.3
    }
    
    # Add financial context to prompt
    enhanced_message = (f"User financial context: {financial_context}\n"
                        f"User question: {user_message}")
    
    response = get_ai_response(session_id, enhanced_message)
    
    resp = jsonify({'response': response})
    resp.set_cookie('session_id', session_id)
    return resp

@app.route('/add_client', methods=['POST'])
def add_client():
    data = load_data()
    new_client = {
        'id': str(uuid.uuid4()),
        'name': request.form['name'],
        'email': request.form['email'],
        'created_at': datetime.now().strftime('%Y-%m-%d')
    }
    data['clients'].append(new_client)
    save_data(data)
    return redirect(url_for('index'))

@app.route('/add_income', methods=['POST'])
def add_income():
    data = load_data()
    new_income = {
        'id': str(uuid.uuid4()),
        'client_id': request.form['client_id'],
        'client_name': next((c['name'] for c in data['clients'] if c['id'] == request.form['client_id']), 'Unknown Client'),
        'description': request.form['description'],
        'amount': float(request.form['amount']),
        'date': request.form['date']
    }
    data['income'].append(new_income)
    save_data(data)
    return redirect(url_for('index'))

@app.route('/generate_invoice', methods=['POST'])
def generate_invoice():
    try:
        data = load_data()
        client_id = request.form['client_id']
        items = request.form.getlist('items[]')
        
        # Validate client exists
        client = next((c for c in data['clients'] if c['id'] == client_id), None)
        if not client:
            return jsonify({'error': 'Client not found'}), 404
            
        # Validate items exist
        valid_items = []
        total_amount = 0
        for item_id in items:
            item = next((i for i in data['income'] if i['id'] == item_id), None)
            if item:
                valid_items.append({
                    'description': item['description'],
                    'amount': item['amount'],
                    'id': item['id']
                })
                total_amount += item['amount']
        
        if not valid_items:
            return jsonify({'error': 'No valid items selected'}), 400
            
        # Create invoice
        new_invoice = {
            'id': str(uuid.uuid4()),
            'client_id': client_id,
            'client_name': client['name'],
            'items': valid_items,
            'amount': total_amount,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'due_date': (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            'status': 'pending'
        }
        
        data['invoices'].append(new_invoice)
        save_data(data)
        
        return jsonify({
            'success': True,
            'invoice_id': new_invoice['id']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        save_data({'clients': [], 'invoices': [], 'income': [], 'expenses': []})
    app.run(debug=True)
